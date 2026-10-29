"""Paired SFT/correct/matched evaluation; defaults to validation."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


@hydra.main(config_path='../../configs', config_name='cure', version_base=None)
def main(cfg: DictConfig):
    from utils.logging_config import configure_logging
    configure_logging(cfg.logging.output_path, cfg.logging.level, cfg.logging.mode)
    if cfg.eval.split == 'test' or cfg.data.domain == 'sports':
        if not cfg.eval.design_frozen or not cfg.eval.recipe_id:
            raise ValueError('Test/Sports evaluation requires a frozen recipe_id and design_frozen=true')
    from transformers import set_seed
    from models.cure.runtime import load_components, make_dataset, provenance, assert_checkpoint_identity, assert_evaluation_preprocessing, evaluation_output_path
    from inference.cure_evaluator import evaluate_cure
    from models.cure.distributed import create_accelerator, main_process_call, init_tracking, evaluation_logs, configure_zero3_model, zero_stage
    accelerator = create_accelerator(cfg, training=False)
    try:
        set_seed(cfg.eval.random_seed)
        model, tokenizer, sid_ids, saved = load_components(cfg, cfg.eval.model_path)
        identity = main_process_call(lambda: provenance(cfg), accelerator)
        assert_checkpoint_identity(saved, identity)
        assert_evaluation_preprocessing(saved, cfg)
        # ZeRO-1/2 only shard optimizer state/gradients; inference has neither.
        if zero_stage(accelerator) == 3:
            model = accelerator.prepare(model)
            configure_zero3_model(accelerator.unwrap_model(model), accelerator)
        dataset = make_dataset(cfg, tokenizer, sid_ids, cfg.eval.split, cfg.eval.sample_size)
        init_tracking(cfg, accelerator, suffix=f"_eval_{cfg.eval.split}")
        result = evaluate_cure(model, dataset, tokenizer.pad_token_id, cfg.eval.batch_size, cfg.eval.num_beams,
                      list(cfg.eval.k_values), cfg.eval.random_seed, dict(cfg.matching.buckets),
                      metadata={**identity, 'model_variant': saved['model_variant'], 'method': saved['method'],
                                'split': cfg.eval.split, 'recipe_id': cfg.eval.recipe_id,
                                'cure_checkpoint': str(Path(cfg.eval.model_path).resolve()),
                                'checkpoint_step': saved['step'], 'checkpoint_epoch': saved['epoch'],
                                'training_seed': saved['config']['train']['seed'],
                                'distributed': {'zero_stage': zero_stage(accelerator),
                                                'mixed_precision': accelerator.mixed_precision,
                                                'model_dtype': cfg.model.dtype,
                                                'training': saved.get('distributed', {})}},
                      output_path=evaluation_output_path(cfg), accelerator=accelerator)
        accelerator.log(evaluation_logs(result, cfg.eval.split), step=saved['step'])
    finally:
        accelerator.end_training()


if __name__ == '__main__':
    main()
