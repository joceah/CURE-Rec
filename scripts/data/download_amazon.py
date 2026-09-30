"""
使用 wget 下载 Amazon Reviews 2023 Beauty 数据集（支持断点续传）。

输出到 data/raw/：
  - Beauty.jsonl          （用户评论，约 371MB）
  - meta_Beauty.jsonl     （商品元数据，约 116MB）

用法：
  conda activate MiniOneRec
  python scripts/data/download_amazon.py
  python scripts/data/download_amazon.py --output_dir /path/to/custom/dir

优势：
  - 支持断点续传（中断后重新运行会继续下载）
  - 多线程下载（如果服务器支持）
  - 更稳定的下载体验
"""

import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Amazon Reviews 2023 官方下载地址
FILES = [
    {
        "name": "Beauty.jsonl",
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Beauty_and_Personal_Care.jsonl.gz",
        "desc": "用户评论数据",
    },
    {
        "name": "meta_Beauty.jsonl",
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz",
        "desc": "商品元数据",
    },
]


def download_with_wget(url: str, output_path: Path) -> bool:
    """使用 wget 下载文件，支持断点续传。"""
    cmd = [
        "wget",
        "-c",  # 断点续传
        "--progress=bar:force:noscroll",  # 显示进度条
        "--tries=0",  # 无限重试
        "--timeout=30",  # 超时 30 秒
        "--waitretry=5",  # 重试等待 5 秒
        "-O", str(output_path),
        url,
    ]

    logger.info(f"开始下载: {url}")
    logger.info(f"保存到: {output_path}")
    logger.info("提示：如果下载中断，重新运行脚本会自动断点续传")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"下载失败: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning("\n下载被中断。重新运行脚本会从断点继续下载。")
        return False


def decompress_gz(gz_path: Path, out_path: Path) -> None:
    """解压 .gz 文件。"""
    import gzip
    import shutil

    logger.info(f"解压: {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"解压完成: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def download_all(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for file_info in FILES:
        dest = out / file_info["name"]
        gz_dest = out / (file_info["name"] + ".gz")

        if dest.exists():
            logger.info(f"✓ 已存在，跳过: {dest}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"下载 {file_info['desc']}: {file_info['name']}")
        logger.info(f"{'='*60}")

        # 下载 .gz
        if not gz_dest.exists() or gz_dest.stat().st_size < 1024:  # 小于 1KB 认为不完整
            success = download_with_wget(file_info["url"], gz_dest)
            if not success:
                logger.error(f"下载失败，请重新运行脚本继续下载")
                return
        else:
            logger.info(f"✓ .gz 文件已存在: {gz_dest} ({gz_dest.stat().st_size / 1024 / 1024:.1f} MB)")

        # 解压
        try:
            decompress_gz(gz_dest, dest)
        except Exception as e:
            logger.error(f"解压失败: {e}")
            logger.error("可能是下载不完整，删除 .gz 文件后重新运行")
            return

        # 删除 .gz 节省空间
        gz_dest.unlink()
        logger.info(f"✓ 已删除压缩包: {gz_dest}")

    logger.info("\n" + "="*60)
    logger.info("✓ 所有文件下载完成！")
    logger.info("="*60)
    logger.info(f"\n目录内容:")
    for f in sorted(out.iterdir()):
        if f.is_file():
            logger.info(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    logger.info("\n下一步运行数据处理：")
    logger.info(
        f"  python scripts/data/run_process.py \\\n"
        f"    --review_path {out}/Beauty.jsonl \\\n"
        f"    --meta_path   {out}/meta_Beauty.jsonl \\\n"
        f"    --output_dir  data/processed/beauty"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="下载 Amazon Reviews 2023 Beauty 数据集（wget 版本）")
    parser.add_argument(
        "--output_dir",
        default="data/raw",
        help="下载目录（默认 data/raw）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_all(args.output_dir)
