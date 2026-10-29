"""Local artifact loading and reproducible provenance (no legacy CURE import)."""

import hashlib
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.prefix_tree import SidPrefixTree
from .baseline import FrozenSFT
from .dataset import CUREDataset
from .model import CUREModel

DECODE_PROTOCOL = 'sid-candidate-logsoftmax-3stage-v1'


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}


def provenance(cfg):
    model_path = Path(cfg.model.llm.sft_model_path)
    weights = sorted(model_path.glob('*.safetensors')) or sorted(model_path.glob('pytorch_model*.bin'))
    if not weights:
        raise ValueError("Frozen SFT checkpoint has no model weights")
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = 'unknown', True
    return {'git_sha': sha, 'git_dirty': dirty, 'decode_protocol': DECODE_PROTOCOL,
            'sft_checkpoint_identity': {'config': file_identity(model_path / 'config.json'),
                                        'weights': [file_identity(p) for p in weights]},
            'tokenizer_identity': file_identity(Path(cfg.model.llm.tokenizer_path) / 'tokenizer.json'),
            'item_index_hash': file_identity(cfg.data.item_index_path)['sha256'],
            'dataset_manifest': file_identity(cfg.data.manifest_path),
            'train_sequences_identity': file_identity(cfg.data.user_sequences_path)}


def load_components(cfg, checkpoint=None):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm.tokenizer_path, local_files_only=cfg.model.llm.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    with open(cfg.data.item_index_path) as f:
        index = json.load(f)
    item_sid_ids = {int(i): tokenizer.convert_tokens_to_ids(sid) for i, sid in index.items()}
    for sid in item_sid_ids.values():
        if len(sid) != 3 or any(i is None or i == tokenizer.unk_token_id for i in sid):
            raise ValueError("SID artifact contains invalid/unknown tokens")
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.model.dtype]
    backbone = AutoModelForCausalLM.from_pretrained(cfg.model.llm.sft_model_path,
                                                   torch_dtype=dtype, local_files_only=cfg.model.llm.local_files_only)
    backbone.config.pad_token_id = tokenizer.pad_token_id
    device = str(cfg.model.device)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline = FrozenSFT(backbone.to(device), SidPrefixTree(item_sid_ids))
    if checkpoint:
        model, metadata = CUREModel.load_checkpoint(checkpoint, baseline)
    else:
        model = CUREModel(baseline, **dict(cfg.reasoner),
                          residual_hidden_dim=cfg.residual.hidden_dim,
                          gate_bias=cfg.residual.gate_bias).to(device)
        metadata = {}
    return model, tokenizer, item_sid_ids, metadata


def verify_frozen_split(cfg, split):
    with open(cfg.data.manifest_path) as f:
        manifest = json.load(f)
    if manifest.get('freeze_status') != 'frozen' or manifest.get('failed_checks'):
        raise ValueError("Dataset manifest must record a successful freeze audit")
    checks = [(cfg.data.user_sequences_path, manifest['processed_artifacts']['user_sequences.json'])]
    if split != 'train':
        checks.extend([
            (cfg.data[f'{split}_path'], manifest['processed_artifacts'][f'{split}.csv']),
            (Path(cfg.data.manifest_path).parent / f'{split}_sample_ids.csv', manifest['fixed_sample_ids'][split])])
    if split == 'test':
        checks.append((cfg.data.valid_path, manifest['processed_artifacts']['valid.csv']))
    for path, frozen in checks:
        if file_identity(path)['sha256'] != frozen['sha256']:
            raise ValueError(f"Artifact no longer matches the frozen manifest: {path}")


def make_dataset(cfg, tokenizer, item_sid_ids, split='train', sample_size=None):
    if split not in ('train', 'valid', 'test'):
        raise ValueError(f"Unknown split: {split}")
    verify_frozen_split(cfg, split)
    preprocessing = cfg.train if split == 'train' else cfg.eval
    return CUREDataset(
        cfg.data.user_sequences_path, item_sid_ids, tokenizer,
        max_history_len=preprocessing.max_history_len, max_seq_len=preprocessing.max_seq_len,
        split_path=None if split == 'train' else cfg.data[f'{split}_path'],
        valid_history_path=cfg.data.valid_path if split == 'test' else None,
        sample_ids_path=str(Path(cfg.data.manifest_path).parent / f'{split}_sample_ids.csv') if split != 'train' else None,
        sample_size=sample_size, seed=cfg.train.seed if split == 'train' else cfg.eval.random_seed)


def assert_checkpoint_identity(saved, current):
    for key in ('decode_protocol', 'sft_checkpoint_identity', 'tokenizer_identity',
                'item_index_hash', 'dataset_manifest', 'train_sequences_identity'):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Checkpoint provenance mismatch: {key}")


def evaluation_output_path(cfg):
    return Path(cfg.eval.output_dir) / f'{cfg.eval.output_name}.json'


def assert_evaluation_preprocessing(saved, cfg):
    # Older checkpoints recorded eval preprocessing only in their train section.
    old_eval = saved['config'].get('eval', {})
    for key in ('max_history_len', 'max_seq_len'):
        expected = old_eval.get(key, saved['config']['train'][key])
        if expected != cfg.eval[key]:
            raise ValueError(f'Evaluation preprocessing differs from checkpoint: {key}')
