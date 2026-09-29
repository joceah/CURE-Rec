"""
Stage 1 数据处理启动脚本。

用法：
  conda activate MiniOneRec
  # 全量数据
  python scripts/data/run_process.py \
    --review_path data/raw/Beauty.jsonl \
    --meta_path   data/raw/meta_Beauty.jsonl \
    --output_dir  data/processed/beauty

  # 只处理 2018-2023 年的数据
  python scripts/data/run_process.py \
    --review_path data/raw/Beauty.jsonl \
    --meta_path   data/raw/meta_Beauty.jsonl \
    --output_dir  data/processed/beauty_2018_2023 \
    --start_date 2018-01-01 \
    --end_date   2023-12-31
"""

import argparse
import logging
import sys
from pathlib import Path

# 把 src 加入 path，无需安装包
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.amazon_process import process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon Reviews Stage 1 数据处理")
    parser.add_argument("--review_path", required=True, help="原始评论 jsonl 路径")
    parser.add_argument("--meta_path", required=True, help="商品元数据 jsonl 路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--k", type=int, default=5, help="k-core 过滤阈值（默认 5）")
    parser.add_argument("--start_date", default=None, help="起始日期（含），格式 YYYY-MM-DD")
    parser.add_argument("--end_date", default=None, help="截止日期（含），格式 YYYY-MM-DD")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        review_path=args.review_path,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        k=args.k,
        start_date=args.start_date,
        end_date=args.end_date,
    )
