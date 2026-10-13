"""Stage 1 数据处理启动脚本。读取 configs/stage1.yaml，同时兼容旧路径参数。"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from data.amazon_process import process

from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon Reviews Stage 1 数据处理")
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "stage1.yaml"),
        help="阶段 1 配置文件",
    )
    parser.add_argument("--domain", help="数据域：beauty 或 sports")
    parser.add_argument("--review_path", help="原始评论 jsonl 路径（覆盖配置）")
    parser.add_argument("--meta_path", help="商品元数据 jsonl 路径（覆盖配置）")
    parser.add_argument("--output_dir", help="输出目录（覆盖配置）")
    parser.add_argument("--k", type=int, help="k-core 过滤阈值（覆盖配置）")
    parser.add_argument("--start_date", help="起始日期（含），格式 YYYY-MM-DD")
    parser.add_argument("--end_date", help="截止日期（含），格式 YYYY-MM-DD")
    parser.add_argument(
        "--all_dates",
        action="store_true",
        help="不应用配置中的日期范围",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    configure_logging(cfg.logging.output_path, cfg.logging.level, cfg.logging.mode)
    domain = args.domain or cfg.domain
    if domain not in cfg.domains:
        raise ValueError(f"配置 {args.config} 中没有数据域: {domain}")

    domain_cfg = cfg.domains[domain]
    explicit_paths = any((args.review_path, args.meta_path, args.output_dir))
    review_path = args.review_path or domain_cfg.review_path
    meta_path = args.meta_path or domain_cfg.meta_path
    output_dir = args.output_dir or domain_cfg.output_dir
    k = args.k if args.k is not None else cfg.k

    if args.all_dates or (explicit_paths and args.start_date is None and args.end_date is None):
        start_date = end_date = None
    else:
        start_date = args.start_date or cfg.get("start_date")
        end_date = args.end_date or cfg.get("end_date")

    logger.info(
        "Stage 1 config=%s domain=%s reviews=%s metadata=%s output=%s k=%s dates=%s..%s",
        args.config,
        domain,
        review_path,
        meta_path,
        output_dir,
        k,
        start_date,
        end_date,
    )
    process(
        review_path=review_path,
        meta_path=meta_path,
        output_dir=output_dir,
        k=k,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
