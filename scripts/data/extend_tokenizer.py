"""从新的 item.index.json 提取 SID tokens，扩展 tokenizer。

用法：
    python scripts/data/extend_tokenizer.py \\
        --base_tokenizer Qwen/Qwen2.5-0.5B-Instruct \\
        --item_index_path data/processed/beauty_2021_2023/item.index.json \\
        --output_dir data/tokenizers/sft_v2_codebook512
"""
import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--item_index_path",
                        default="data/processed/beauty_2021_2023/item.index.json")
    parser.add_argument("--output_dir",
                        default="data/tokenizers/sft_v2_codebook512")
    args = parser.parse_args()

    print("=" * 70)
    print("扩展 Tokenizer（添加 SID special tokens）")
    print("=" * 70)

    # 1. 加载 base tokenizer
    print(f"\n[1/4] 加载 base tokenizer: {args.base_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    print(f"  原始 vocab_size: {tokenizer.vocab_size}")
    print(f"  原始 len(tokenizer): {len(tokenizer)}")

    # 2. 从 item.index.json 提取所有 SID tokens
    print(f"\n[2/4] 从 item.index.json 提取 SID tokens: {args.item_index_path}")
    with open(args.item_index_path, "r", encoding="utf-8") as f:
        item_index = json.load(f)

    all_sid_tokens = set()
    for sid_list in item_index.values():
        all_sid_tokens.update(sid_list)

    # 排序（按层 + 数字）
    sid_tokens_sorted = sorted(
        all_sid_tokens,
        key=lambda t: (t[1], int(t.split('_')[1].rstrip('>')))
    )
    print(f"  提取到 {len(sid_tokens_sorted)} 个唯一 SID token")

    # 按层统计
    from collections import defaultdict
    layer_counts = defaultdict(int)
    for t in sid_tokens_sorted:
        layer = t[1]  # 'a', 'b', 'c'
        layer_counts[layer] += 1
    print(f"  各层分布: {dict(layer_counts)}")
    print(f"  样例: {sid_tokens_sorted[:5]} ... {sid_tokens_sorted[-5:]}")

    # 3. 扩展 tokenizer
    print(f"\n[3/4] 添加 SID tokens 到 tokenizer")
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": sid_tokens_sorted})
    print(f"  新增 token 数: {num_added}")
    print(f"  扩展后 vocab_size: {tokenizer.vocab_size}")
    print(f"  扩展后 len(tokenizer): {len(tokenizer)}")

    # 验证
    print(f"\n  验证：随机抽查 5 个 token 的 ID")
    random.seed(42)
    samples = random.sample(sid_tokens_sorted, min(5, len(sid_tokens_sorted)))
    for token in samples:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded = tokenizer.convert_ids_to_tokens(token_id)
        print(f"    {token} → {token_id} → {decoded} {'✓' if decoded == token else '✗'}")

    # 4. 保存
    print(f"\n[4/4] 保存到: {args.output_dir}")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    # 修复 added_tokens_decoder（transformers 保存时的 bug）
    config_path = output_path / "tokenizer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 如果缺少 added_tokens_decoder，手动生成
    if "added_tokens_decoder" not in config or config["added_tokens_decoder"] is None:
        print(f"  修复 tokenizer_config.json（补充 added_tokens_decoder）")
        added_tokens_decoder = {}
        for token in sid_tokens_sorted:
            token_id = tokenizer.convert_tokens_to_ids(token)
            added_tokens_decoder[str(token_id)] = {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
        config["added_tokens_decoder"] = added_tokens_decoder
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  保存完成！")

    # 同时保存一份 token 列表到 JSON（用于后续验证）
    token_list_path = output_path / "sid_tokens.json"
    with open(token_list_path, "w", encoding="utf-8") as f:
        json.dump(sid_tokens_sorted, f, indent=2, ensure_ascii=False)
    print(f"  SID token 列表已保存到: {token_list_path}")

    print("\n" + "=" * 70)
    print("✅ Tokenizer 扩展完成！")
    print("=" * 70)
    print(f"\n下一步：用新 tokenizer 重训 SFT")
    print(f"  1. 修改 configs/model/llm/default.yaml:")
    print(f"     base_model: Qwen/Qwen2.5-0.5B-Instruct")
    print(f"     tokenizer_path: {args.output_dir}")
    print(f"  2. 运行 SFT 训练:")
    print(f"     accelerate launch --config_file configs/deepspeed/zero2_opt.yaml \\")
    print(f"       scripts/train/run_sft.py")


if __name__ == "__main__":
    main()
