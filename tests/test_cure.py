"""CURE invariants; tiny real causal Transformer, plus opt-in frozen Beauty integration.

PYTHONPATH=src pytest -q tests/test_cure.py
CURE_REAL_SFT=1 PYTHONPATH=src pytest -q tests/test_cure.py -k real_sft
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from inference.beam_search import generate_topk_sids
from inference.cure_evaluator import evaluate_cure, generate_cure_sids
from inference.prefix_tree import SidPrefixTree, ConstrainedLogitsProcessor
from models.cure import CUREModel, FrozenSFT, cure_loss, score_target
from models.cure.dataset import CUREDataset, collate_cure, make_loader, to_device
from models.cure.losses import safe_kl
from models.cure.matching import bucketize, match_donors, prepare_matching_cache


SID = {0: [10, 20, 30], 1: [10, 20, 31], 2: [10, 21, 30],
       3: [11, 20, 30], 4: [11, 21, 31], 5: [11, 21, 32]}


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [1, 2] if text == '用户历史：' else [3, 4]


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'sequences.json'
    path.write_text(json.dumps({'0': [0, 1, 2, 3], '1': [4, 2, 1], '2': [3, 0, 5]}))
    split = tmp_path / 'valid.csv'
    split.write_text('user_id,item_id\n0,4\n1,3\n2,1\n')
    tokenizer = TinyTokenizer()
    dataset = CUREDataset(path, SID, tokenizer, split_path=split)
    return dataset, tokenizer, path, split


@pytest.fixture
def model():
    torch.manual_seed(11)
    backbone = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        attention_dropout=0., pad_token_id=0, eos_token_id=2, bos_token_id=1))
    return CUREModel(FrozenSFT(backbone, SidPrefixTree(SID)), hidden_dim=16,
                     num_heads=2, dropout=0., residual_hidden_dim=24).eval()


def test_baseline_recovery_and_prefix_parity(model, data):
    dataset, tokenizer, *_ = data
    batch = collate_cure([dataset[0], dataset[1]], tokenizer.pad_token_id)
    features = model.baseline.prepare(batch)
    base = score_target(model.baseline, model, batch, 'baseline', features=features)
    disabled = model.score(features, model.encode(features), disabled=True)
    assert torch.equal(base.stage_log_probs, disabled.stage_log_probs)
    for i, steps in enumerate(features.steps):
        prompt_length = batch['prompt_length'][i].item()
        prompt = batch['input_ids'][i, :prompt_length].tolist()
        for t, step in enumerate(steps):
            prefix = features.targets[i, :t].tolist()
            assert set(step.candidate_ids.tolist()) == model.baseline.tree.get_allowed_tokens(tuple(prefix))
            processor = ConstrainedLogitsProcessor(model.baseline.tree, torch.tensor([prompt_length]), pad_token_id=0)
            masked = processor(torch.tensor([prompt + prefix]), torch.zeros(1, 64))
            assert torch.equal(torch.isfinite(masked[0]).nonzero().flatten(), step.candidate_ids)
            assert torch.equal(disabled.candidate_logits[i][t], step.candidate_logits)
            assert torch.equal(disabled.stage_distributions[i][t], step.candidate_logits.log_softmax(-1))
    assert base.stage_log_probs.shape == (2, 3)  # no EOS


def test_target_leakage_and_generation_stage_parity(model, data):
    dataset, tokenizer, *_ = data
    sample = dataset[0]
    batch = collate_cure([sample], 0)
    features = model.baseline.prepare(batch)
    z = model.encode(features)
    altered = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    length = batch['prompt_length'][0].item()
    altered['input_ids'][0, length:length + 3] = torch.tensor(SID[5])
    changed = model.baseline.prepare(altered)
    assert torch.equal(features.decision, changed.decision)
    assert torch.equal(z, model.encode(changed))
    scores = model.score(features, z)
    prompt = batch['input_ids'][0, :length].tolist()
    for t in range(3):
        step = model.baseline.step(prompt, features.targets[0, :t].tolist())
        actual, _, _ = model.candidate_step(step, z[0])
        torch.testing.assert_close(actual, scores.candidate_logits[0][t], atol=1e-6, rtol=1e-5)


def test_frozen_and_residual_gradients(model, data):
    dataset, *_ = data
    model.train()
    assert not model.baseline.model.training
    batch = collate_cure([dataset[0], dataset[1]], 0)
    outputs = model(batch, torch.tensor([1, 0]))
    loss = cure_loss(outputs, variant='cure_full')['loss']
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.baseline.parameters())
    for module in (model.reasoner, model.residual.gate, model.residual.mlp, model.residual.projection):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    assert outputs['correct'].sequence_score.requires_grad
    assert not outputs['baseline'].sequence_score.requires_grad


def test_matching_nonself_and_relaxation():
    buckets = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 2, 1],
                            [2, 0, 2], [3, 0, 2], [4, 0, 3]])
    a, levels = match_donors(buckets, torch.Generator().manual_seed(9))
    b, _ = match_donors(buckets, torch.Generator().manual_seed(9))
    assert torch.equal(a, b)
    assert (a != torch.arange(len(a))).all()
    assert levels.tolist() == [0, 0, 1, 1, 2, 2, 3]
    with pytest.raises(ValueError, match='at least two'):
        match_donors(buckets[:1])
    with pytest.raises(ValueError, match='distinct'):
        match_donors(buckets[:2], sample_ids=['same', 'same'])


def test_loss_sanity_and_kl_direction():
    def scored(p):
        lp = torch.tensor(p, dtype=torch.float64).log()
        return SimpleNamespace(sequence_score=lp[0:1], stage_distributions=[[lp] * 3])
    base, correct, wrong = scored([.5, .5]), scored([.8, .2]), scored([.5, .5])
    outputs = dict(baseline=base, correct=correct, matched=wrong)
    losses = cure_loss(outputs, variant='cure_full', margin_gain=.1, margin_personal=.1)
    assert losses['gain'].item() == 0
    assert losses['personal'].item() == 0
    assert losses['safe'].item() == 0
    assert losses['rec'].item() == pytest.approx(-torch.tensor(.8).log().item())
    unsafe = scored([.1, .9])
    expected = .5 * (__import__('math').log(.5 / .1) + __import__('math').log(.5 / .9))
    assert safe_kl(base, unsafe).item() == pytest.approx(expected)
    worse = dict(outputs, correct=scored([.4, .6]))
    assert cure_loss(worse, variant='cure_g')['gain'] > 0
    for variant in ('residual_ce', 'cure_g'):
        assert torch.isfinite(cure_loss({k: v for k, v in outputs.items() if k != 'matched'}, variant=variant)['loss'])
    with pytest.raises(KeyError):
        cure_loss(outputs={k: v for k, v in outputs.items() if k != 'matched'}, variant='cure_gp')


def test_checkpoint_roundtrip(model, data, tmp_path):
    dataset, *_ = data
    batch = collate_cure([dataset[0], dataset[1]], 0)
    expected = model(batch, torch.tensor([1, 0]))['correct'].stage_log_probs
    path = tmp_path / 'model.pt'
    model.save_checkpoint(path, {'model_variant': 'cure_full'})
    raw = torch.load(path, weights_only=True)
    assert set(raw) == {'format', 'architecture', 'reasoner', 'residual', 'metadata'}
    restored, metadata = CUREModel.load_checkpoint(path, model.baseline)
    restored.eval()
    assert metadata['model_variant'] == 'cure_full'
    assert torch.equal(expected, restored(batch, torch.tensor([1, 0]))['correct'].stage_log_probs)


def test_candidate_normalized_generation_parity(model, data):
    dataset, tokenizer, *_ = data
    prompts = [sample['input_ids'][:sample['prompt_length']].tolist() for sample in (dataset[0], dataset[1])]
    reference = generate_topk_sids(model.baseline.model, tokenizer, prompts, model.baseline.tree,
                                   num_beams=4, device='cpu', candidate_normalized=True)
    actual = [generate_cure_sids(model, prompt, num_beams=4, disabled=True) for prompt in prompts]
    assert reference == actual
    for predicted in actual:
        assert all(list(sid) in list(SID.values()) for sid in predicted)


def test_dataset_protocol_and_truncation(data):
    dataset, tokenizer, path, split = data
    test_split = split.parent / 'test.csv'
    test_split.write_text('user_id,item_id\n0,5\n1,5\n2,4\n')
    test = CUREDataset(path, SID, tokenizer, split_path=test_split, valid_history_path=split,
                       max_seq_len=13)
    sample = test[0]
    assert sample['history_len'] == 2
    positions = sample['history_positions']
    assert torch.equal(sample['input_ids'][positions[-1]], torch.tensor(SID[4]))
    assert (positions < sample['prompt_length']).all()
    assert sample['labels'][-3:].tolist() == SID[5]
    assert sample['target_popularity'] == 1  # train only
    train = CUREDataset(path, SID, tokenizer)
    assert len(train) == 7
    assert train[0]['input_ids'][-3:].tolist() == SID[1]


def test_cache_identity_and_scores(model, data, tmp_path):
    dataset, *_ = data
    loader = make_loader(dataset, 2, 0)
    path = tmp_path / 'difficulty.pt'
    cache = prepare_matching_cache(model, loader, path, {'sft': 'tiny'})
    loaded = prepare_matching_cache(model, loader, path, {'sft': 'tiny'})
    assert torch.equal(cache['scores'], loaded['scores'])
    assert cache['buckets'].shape == (3, 3)
    with pytest.raises(ValueError, match='Stale'):
        prepare_matching_cache(model, loader, path, {'sft': 'changed'})


def test_evaluation_outputs_and_singleton_tail(model, data, tmp_path):
    dataset, *_ = data
    path = tmp_path / 'metrics.json'
    result = evaluate_cure(model, dataset, 0, batch_size=2, num_beams=4, k_values=[1, 4],
                           metadata={'model_variant': 'residual_ce'}, output_path=path)
    assert path.exists()
    assert len(result['sample_ids']) == 3
    assert all(a != b for a, b in zip(result['sample_ids'], result['donor_sample_ids']))
    for k in ('HR@1', 'HR@4'):
        assert result['RG'][k] == pytest.approx(result['PG'][k] - result['FD'][k])
    assert result['mean_safe_KL'] >= -1e-7
    assert model.training is False


@pytest.mark.skipif(os.environ.get('CURE_REAL_SFT') != '1', reason='Opt-in local Beauty SFT integration')
def test_real_sft_invariants(tmp_path):
    from omegaconf import OmegaConf
    from models.cure.runtime import load_components, make_dataset
    cfg = OmegaConf.load('configs/cure.yaml')
    model, tokenizer, sids, _ = load_components(cfg)
    model.eval()
    dataset = make_dataset(cfg, tokenizer, sids, 'valid', 2)
    batch = to_device(collate_cure([dataset[0], dataset[1]], tokenizer.pad_token_id), model.baseline.device)
    with torch.no_grad():
        features = model.baseline.prepare(batch)
        z = model.encode(features)
        base = model.score(features, disabled=True)
        off = model.score(features, z, disabled=True)
        assert torch.equal(base.stage_log_probs, off.stage_log_probs)
        changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for i in range(2):
            p = batch['prompt_length'][i]
            changed['input_ids'][i, p:p + 3] = features.targets[1 - i]
        z_changed = model.encode(model.baseline.prepare(changed))
        assert torch.equal(z, z_changed)
        for i in range(2):
            prompt = batch['input_ids'][i, :batch['prompt_length'][i]].tolist()
            for t in range(3):
                step = model.baseline.step(prompt, features.targets[i, :t].tolist())
                # BF16 kernels for batched vs single-token/cache GEMMs can round differently.
                torch.testing.assert_close(step.candidate_logits, base.candidate_logits[i][t], atol=.15, rtol=.01)
            actual = generate_cure_sids(model, prompt, num_beams=10, disabled=True)
            reference = generate_topk_sids(model.baseline.model, tokenizer, [prompt], model.baseline.tree,
                                           num_beams=10, device=model.baseline.device, candidate_normalized=True)[0]
            assert actual == reference
    outputs = model(batch, torch.tensor([1, 0], device=model.baseline.device))
    cure_loss(outputs, variant='cure_full')['loss'].backward()
    assert all(p.grad is None for p in model.baseline.parameters())
    assert model.reasoner.query.weight.grad.abs().sum() > 0
    model.save_checkpoint(tmp_path / 'real.pt')
    restored, _ = CUREModel.load_checkpoint(tmp_path / 'real.pt', model.baseline)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(outputs['correct'].stage_log_probs,
                           restored(batch)['correct'].stage_log_probs)


def test_actual_generation_processor_logits(model, data):
    from inference.cure_evaluator import _CURELogitsProcessor
    dataset, *_ = data
    batch = collate_cure([dataset[0]], 0)
    features = model.baseline.prepare(batch)
    latent = model.encode(features)
    expected = model.score(features, latent)
    prompt = batch['input_ids'][0, :batch['prompt_length'][0]].tolist()
    for t in range(3):
        prefix = features.targets[0, :t].tolist()
        step = model.baseline.step(prompt, prefix)
        processor = _CURELogitsProcessor(model, len(prompt), latent[0], {'last': step.hidden[None]}, False)
        actual = processor(torch.tensor([prompt + prefix]), step.logits[None].log_softmax(-1))[0]
        allowed = step.candidate_ids
        assert torch.isfinite(actual).sum() == len(allowed)
        torch.testing.assert_close(actual[allowed].log_softmax(-1), expected.stage_distributions[0][t], atol=1e-6, rtol=1e-5)


def test_training_variants_and_singleton_tail(model, data, tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from models.cure import trainer
    dataset, tokenizer, path, _ = data
    train = CUREDataset(path, SID, tokenizer, sample_size=5)
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.model.device = 'cpu'
    cfg.distributed.use_deepspeed = False
    cfg.train.epochs = 1
    cfg.train.per_device_batch_size = 2
    cfg.train.gradient_accumulation_steps = 2
    cfg.train.max_steps = 1
    cfg.eval.num_beams = 10
    cfg.eval.k_values = [1, 5, 10]
    cfg.matching.cache_path = str(tmp_path / 'difficulty.pt')
    monkeypatch.setattr(trainer, 'load_components', lambda *args: (model, tokenizer, SID, {}))
    monkeypatch.setattr(trainer, 'make_dataset', lambda *args, **kwargs: train if len(args) == 3 else dataset)
    identity = {key: 'tiny' for key in ('sft_checkpoint_identity', 'tokenizer_identity', 'item_index_hash',
                                       'dataset_manifest', 'train_sequences_identity', 'decode_protocol')}
    monkeypatch.setattr(trainer, 'provenance', lambda cfg: identity)
    for variant in ('residual_ce', 'cure_g', 'cure_gp', 'cure_full'):
        cfg.method.variant = variant
        cfg.train.output_dir = str(tmp_path / variant)
        result = trainer.run_cure(cfg)
        assert result['steps'] == 1
        checkpoint = torch.load(result['best_checkpoint'], weights_only=True)
        assert checkpoint['metadata']['model_variant'] == variant
        assert checkpoint['metadata']['step'] == 1
        assert len(checkpoint['metadata']['validation_selection']) == 4


def test_invalid_donor_indices(model, data):
    dataset, *_ = data
    batch = collate_cure([dataset[0], dataset[1]], 0)
    for indices in (torch.tensor([0, 0]), torch.tensor([-1, 0]), torch.tensor([2, 0])):
        with pytest.raises(ValueError, match='non-self'):
            model(batch, indices)


def test_evaluation_config_paths_and_seed(data, tmp_path, monkeypatch):
    """Eval path/history/seed settings take effect independently of training."""
    from omegaconf import OmegaConf
    from models.cure import runtime
    _, tokenizer, _, _ = data
    cfg = OmegaConf.load('configs/cure.yaml')
    sequences = tmp_path / 'custom_sequences.json'
    sequences.write_text(json.dumps({str(i): [0, 1, 2, 3] for i in range(30)}))
    split = tmp_path / 'custom_validation.csv'
    split.write_text('user_id,item_id\n' + ''.join(f'{i},{i % 6}\n' for i in range(30)))
    identities = tmp_path / 'valid_sample_ids.csv'
    identities.write_text('user_id,target_item_id,sample_id\n' + ''.join(f'{i},{i % 6},valid:{i}:{i % 6}\n' for i in range(30)))
    cfg.data.user_sequences_path = str(sequences)
    cfg.data.valid_path = str(split)
    cfg.data.manifest_path = str(tmp_path / 'manifest.json')
    cfg.eval.max_history_len = 2
    cfg.eval.random_seed = 17
    monkeypatch.setattr(runtime, 'verify_frozen_split', lambda *a: None)
    a = runtime.make_dataset(cfg, tokenizer, SID, 'valid', 5)
    cfg.train.seed = 999
    b = runtime.make_dataset(cfg, tokenizer, SID, 'valid', 5)
    assert a.samples == b.samples
    assert a[0]['history_len'] == 2
    cfg.eval.random_seed = 19
    c = runtime.make_dataset(cfg, tokenizer, SID, 'valid', 5)
    assert a.samples != c.samples


def test_legacy_checkpoint_preprocessing_config():
    from omegaconf import OmegaConf
    from models.cure.runtime import assert_evaluation_preprocessing
    cfg = OmegaConf.load('configs/cure.yaml')
    legacy = {'config': {'train': {'max_history_len': 20, 'max_seq_len': 512},
                         'eval': {'checkpoint': 'old/best.pt'}}}
    assert_evaluation_preprocessing(legacy, cfg)
    cfg.eval.max_history_len = 10
    with pytest.raises(ValueError, match='max_history_len'):
        assert_evaluation_preprocessing(legacy, cfg)
    current = {'config': {'train': legacy['config']['train'],
                          'eval': {'max_history_len': 10, 'max_seq_len': 512}}}
    assert_evaluation_preprocessing(current, cfg)


def test_evaluation_cli_uses_canonical_config(model, data, tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from models.cure import runtime
    from scripts.eval.run_cure_eval import main
    import utils.logging_config
    dataset, tokenizer, *_ = data
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.model.device = 'cpu'
    cfg.distributed.use_deepspeed = False
    cfg.train.seed = 999
    cfg.eval.random_seed = 17
    cfg.eval.model_path = str(tmp_path / 'selected_branch.pt')
    cfg.eval.output_dir = str(tmp_path / 'selected_results')
    cfg.eval.output_name = 'validation_selected'
    cfg.eval.num_beams = 4
    cfg.eval.k_values = [1, 4]
    identity = {key: 'tiny' for key in ('decode_protocol', 'sft_checkpoint_identity', 'tokenizer_identity',
                                       'item_index_hash', 'dataset_manifest', 'train_sequences_identity')}
    # Older checkpoint metadata has no new eval preprocessing keys.
    saved = {**identity, 'config': {'train': {'max_history_len': 20, 'max_seq_len': 512, 'seed': 42}},
             'model_variant': 'residual_ce', 'method': {'variant': 'residual_ce'}, 'step': 1, 'epoch': 1}
    selected_paths = []

    def load(cfg, checkpoint):
        selected_paths.append(checkpoint)
        return model, tokenizer, SID, saved

    monkeypatch.setattr(runtime, 'load_components', load)
    monkeypatch.setattr(runtime, 'provenance', lambda cfg: identity)
    monkeypatch.setattr(runtime, 'make_dataset', lambda *args: dataset)
    monkeypatch.setattr(utils.logging_config, 'configure_logging', lambda *args: None)
    main.__wrapped__(cfg)
    assert selected_paths == [str(tmp_path / 'selected_branch.pt')]
    result = json.loads((tmp_path / 'selected_results' / 'validation_selected.json').read_text())
    assert result['seed'] == 17 and result['training_seed'] == 42


@pytest.mark.parametrize('enabled,expected_steps', [(True, 3), (False, 4)])
def test_step_validation_and_early_stop(model, data, tmp_path, monkeypatch, enabled, expected_steps):
    from omegaconf import OmegaConf
    from models.cure import trainer
    dataset, tokenizer, path, _ = data
    train = CUREDataset(path, SID, tokenizer)
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.model.device = 'cpu'
    cfg.distributed.use_deepspeed = False
    cfg.train.epochs = 3
    cfg.train.per_device_batch_size = 2
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.max_steps = 4
    cfg.train.eval_steps = 1
    cfg.train.early_stop.enabled = enabled
    cfg.train.early_stop.patience_steps = 2
    cfg.train.early_stop.min_delta = .01
    cfg.train.output_dir = str(tmp_path / 'run')
    cfg.eval.num_beams = 10
    cfg.eval.k_values = [1, 5, 10]
    monkeypatch.setattr(trainer, 'load_components', lambda *a: (model, tokenizer, SID, {}))
    monkeypatch.setattr(trainer, 'make_dataset', lambda *a, **kw: train if len(a) == 3 else dataset)
    monkeypatch.setattr(trainer, 'provenance', lambda cfg: {})
    calls = []

    def validate(*args, **kwargs):
        step = kwargs['metadata']['step']
        calls.append(step)
        # Tiny improvements still select checkpoints, but do not reset patience.
        metrics = {'NDCG@10': .1 + step * .00001, 'HR@10': .2}
        return {'metrics': {mode: metrics for mode in ('baseline', 'correct', 'matched')},
                'RG': {'HR@10': 0.}, 'PG': {'HR@10': 0.}, 'FD': {'HR@10': 0.},
                'mean_target_score_gain': .01}

    monkeypatch.setattr(trainer, 'evaluate_cure', validate)
    result = trainer.run_cure(cfg)
    assert result['steps'] == expected_steps
    assert calls == list(range(1, expected_steps + 1))  # no duplicate epoch-end validation
    assert result['stopped_early'] == enabled
    assert result['early_stop']['best_step'] == 1
    assert result['best_checkpoint_step'] == expected_steps
    assert result['stop_reason'] == ('early_stop' if enabled else 'max_steps')
    saved = json.loads((tmp_path / 'run' / 'training_summary.json').read_text())
    assert saved['steps'] == expected_steps


@pytest.mark.parametrize('variant', ['residual_ce', 'cure_full'])
def test_batched_scores_losses_and_gradients_match_loop(model, data, variant):
    import copy
    dataset, *_ = data
    batch = collate_cure([dataset[i] for i in range(len(dataset))], 0)
    features = model.baseline.prepare(batch)
    reference = copy.deepcopy(model)
    donor = torch.tensor([1, 2, 0])

    def loop_score(model, latent=None, disabled=False):
        log_probs, distributions, gates, norms = [], [], [], []
        for i, steps in enumerate(features.steps):
            row_lp, row_dist, row_gates, row_norms = [], [], [], []
            for t, step in enumerate(steps):
                logits, gate, correction = model.candidate_step(step, None if latent is None else latent[i], disabled)
                lp = logits.log_softmax(-1)
                index = (step.candidate_ids == features.targets[i, t]).nonzero().flatten()[0]
                row_lp.append(lp[index]); row_dist.append(lp)
                row_gates.append(gate); row_norms.append(correction.norm())
            log_probs.append(torch.stack(row_lp)); distributions.append(row_dist)
            gates.append(torch.stack(row_gates)); norms.append(torch.stack(row_norms))
        stage = torch.stack(log_probs)
        return SimpleNamespace(sequence_score=stage.mean(-1), stage_log_probs=stage,
                               stage_distributions=distributions, gates=torch.stack(gates),
                               correction_norms=torch.stack(norms))

    def outputs(model, score):
        z = model.encode(features)
        return {'baseline': score(model, disabled=True), 'correct': score(model, z),
                'matched': score(model, z[donor])}

    actual = outputs(model, lambda m, latent=None, disabled=False: m.score(features, latent, disabled))
    expected = outputs(reference, loop_score)
    assert (~actual['correct'].candidate_mask).any()  # exercises ragged padding
    for mode in actual:
        for field in ('sequence_score', 'stage_log_probs', 'gates', 'correction_norms'):
            torch.testing.assert_close(getattr(actual[mode], field), getattr(expected[mode], field), atol=1e-6, rtol=1e-5)
    actual_loss, expected_loss = cure_loss(actual, variant=variant), cure_loss(expected, variant=variant)
    for key in actual_loss:
        torch.testing.assert_close(actual_loss[key], expected_loss[key], atol=1e-6, rtol=1e-5)
    actual_loss['loss'].backward(); expected_loss['loss'].backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_output_overwrite_is_explicit_and_clears_only_run_artifacts(tmp_path):
    from omegaconf import OmegaConf
    from models.cure.trainer import _initialize_output
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.train.output_dir = str(tmp_path)
    artifacts = ['best.pt', 'last.pt', 'best_validation.json', 'training_summary.json',
                 'valid_step_100.json', 'valid_epoch_1.json']
    for name in artifacts + ['config.yaml', 'notes.txt']:
        (tmp_path / name).write_text('old')
    wandb = tmp_path / 'wandb'
    wandb.mkdir()
    (wandb / 'history').write_text('keep')
    with pytest.raises(FileExistsError, match='train.overwrite_output_dir=true'):
        _initialize_output(cfg)
    assert all((tmp_path / name).read_text() == 'old' for name in artifacts)
    cfg.train.overwrite_output_dir = True
    _initialize_output(cfg)
    assert all(not (tmp_path / name).exists() for name in artifacts)
    assert (tmp_path / 'notes.txt').read_text() == 'old'
    assert (wandb / 'history').read_text() == 'keep'
    assert OmegaConf.load(tmp_path / 'config.yaml').train.overwrite_output_dir


def test_in_place_init_is_loaded_before_overwrite(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from models.cure import trainer
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.model.device = 'cpu'
    cfg.distributed.use_deepspeed = False
    cfg.train.output_dir = str(tmp_path)
    cfg.train.overwrite_output_dir = True
    cfg.train.init_checkpoint = str(tmp_path / 'best.pt')
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_text('initial weights')
    calls = []

    def load(*args):
        calls.append(checkpoint.read_text())
        return None, None, None, {}

    def after_cleanup(cfg):
        assert not checkpoint.exists()
        assert calls == ['initial weights']
        raise RuntimeError('verified initialization order')

    monkeypatch.setattr(trainer, 'load_components', load)
    monkeypatch.setattr(trainer, 'provenance', after_cleanup)
    with pytest.raises(RuntimeError, match='verified initialization order'):
        trainer.run_cure(cfg)
