"""GT Rank分布对比实验

对比 SFT baseline 和 CoT对齐模型（如1K CoT）：
1. 对每个test样本，计算 GT item 在生成的Top-K中的rank
2. 统计rank分布（Top-1, 2-5, 6-10, 11-20, 21-50, >50）
3. 分析：CoT模型是否把中间rank推到Top-1（同时也把它们推到>50）

用法:
  python scripts/analysis/compare_gt_rank.py \
    --baseline_ckpt data/checkpoints/sft/merged \
    --cot_ckpt data/checkpoints/cot_align_1000/model \
    --tokenizer_baseline data/checkpoints/sft/tokenizer \
    --tokenizer_cot data/checkpoints/cot_align_1000/tokenizer \
    --dataset beauty \
    --sample_size 1000 \
    --num_beams 100 \
    --output data/analysis/gt_rank_comparison.json
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inference.beam_search import generate_topk_sids
from inference.prefix_tree import SidPrefixTree


def load_test_samples(
    test_csv_path: str,
    user_sequences_path: str,
    item_sid_ids: dict[int, list[int]],
    max_history_len: int,
    sample_size: int,
    seed: int = 42,
):
    with open(user_sequences_path) as f:
        user_sequences = json.load(f)
    user_sequences = {int(k): v for k, v in user_sequences.items()}

    samples = []
    with open(test_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            if uid not in user_sequences or len(user_sequences[uid]) == 0:
                continue
            if iid not in item_sid_ids:
                continue
            history = user_sequences[uid][-max_history_len:]
            samples.append({
                "user_id": uid,
                "history": history,
                "target_sid": tuple(item_sid_ids[iid]),
            })

    if sample_size < len(samples):
        random.seed(seed)
        samples = random.sample(samples, sample_size)
    return samples


def build_prompt(history, item_sid_ids, tokenizer):
    prefix_ids = tokenizer.encode("用户历史：", add_special_tokens=False)
    history_ids = []
    for iid in history:
        if iid in item_sid_ids:
            history_ids.extend(item_sid_ids[iid])
    suffix_ids = tokenizer.encode(" 请推荐下一个商品：", add_special_tokens=False)
    return prefix_ids + history_ids + suffix_ids


def compute_gt_ranks(model, tokenizer, samples, prefix_tree, item_sid_ids, num_beams, batch_size, device):
    """对每个样本计算GT在Top-K中的rank（1-indexed），未命中记为 num_beams+1"""
    gt_ranks = []

    n = len(samples)
    for i in range(0, n, batch_size):
        batch = samples[i:i+batch_size]
        prompts = [build_prompt(s["history"], item_sid_ids, tokenizer) for s in batch]

        topk_sids = generate_topk_sids(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            prefix_tree=prefix_tree,
            num_beams=num_beams,
            num_sid_layers=3,
            device=device,
        )

        for j, s in enumerate(batch):
            target = s["target_sid"]
            rank = num_beams + 1  # 未命中
            for k, pred in enumerate(topk_sids[j]):
                if pred == target:
                    rank = k + 1
                    break
            gt_ranks.append(rank)

        if (i // batch_size + 1) % 20 == 0:
            hit_rate = sum(1 for r in gt_ranks if r <= num_beams) / len(gt_ranks)
            print(f"  [{i+batch_size}/{n}] hit@{num_beams}={hit_rate:.3f}")

    return gt_ranks


def bucket_ranks(ranks, buckets=(1, 5, 10, 20, 50, 100)):
    """分桶统计"""
    counter = Counter()
    prev = 0
    for b in buckets:
        counter[f"Top-{prev+1}~{b}" if prev != 0 else f"Top-1"] = sum(1 for r in ranks if prev < r <= b)
        prev = b
    counter[f">{buckets[-1]}"] = sum(1 for r in ranks if r > buckets[-1])
    return dict(counter)


def load_model(ckpt_path, tokenizer_path, device):
    print(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"加载 model: {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--cot_ckpt", required=True)
    parser.add_argument("--tokenizer_baseline", required=True)
    parser.add_argument("--tokenizer_cot", required=True)
    parser.add_argument("--dataset", default="beauty", choices=["beauty", "sports"])
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--num_beams", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_history_len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_dir = f"data/processed/{args.dataset}_2021_2023"
    device = torch.device("cuda:0")

    # 使用 baseline tokenizer 加载item_index
    tok_baseline = AutoTokenizer.from_pretrained(args.tokenizer_baseline, trust_remote_code=True)
    if tok_baseline.pad_token_id is None:
        tok_baseline.pad_token_id = tok_baseline.eos_token_id
    with open(f"{data_dir}/item.index.json") as f:
        item_index = json.load(f)
    item_sid_ids = {int(k): tok_baseline.convert_tokens_to_ids(v) for k, v in item_index.items()}
    prefix_tree = SidPrefixTree(item_sid_ids)

    # 加载样本
    samples = load_test_samples(
        test_csv_path=f"{data_dir}/test.csv",
        user_sequences_path=f"{data_dir}/user_sequences.json",
        item_sid_ids=item_sid_ids,
        max_history_len=args.max_history_len,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"样本数: {len(samples)}")

    results = {"config": vars(args), "n_samples": len(samples)}

    # ========== Baseline ==========
    print("\n===== Baseline =====")
    t0 = time.time()
    model_b, tok_b = load_model(args.baseline_ckpt, args.tokenizer_baseline, device)
    ranks_baseline = compute_gt_ranks(
        model_b, tok_b, samples, prefix_tree, item_sid_ids,
        num_beams=args.num_beams, batch_size=args.batch_size, device=device,
    )
    del model_b
    torch.cuda.empty_cache()
    print(f"耗时: {time.time()-t0:.1f}s")

    # ========== CoT ==========
    print("\n===== CoT =====")
    t0 = time.time()
    # CoT tokenizer可能不同（若添加了新special token），需要单独加载
    # 但item_sid_ids保持一致（因为SID token是相同的）
    tok_cot = AutoTokenizer.from_pretrained(args.tokenizer_cot, trust_remote_code=True)
    if tok_cot.pad_token_id is None:
        tok_cot.pad_token_id = tok_cot.eos_token_id
    # 用CoT tokenizer的id重新映射
    item_sid_ids_cot = {int(k): tok_cot.convert_tokens_to_ids(v) for k, v in item_index.items()}
    prefix_tree_cot = SidPrefixTree(item_sid_ids_cot)
    # samples中的target_sid需要重新计算
    samples_cot = []
    for s in samples:
        # 找回原item_id (通过反查)
        # 简化：直接用tokenizer_cot重新构造prompt和target
        samples_cot.append({
            "user_id": s["user_id"],
            "history": s["history"],
            # target_sid：先反查原tokenizer得item_id_str，再用cot tokenizer查id
            # 但因为item_index是一样的，只是tokenizer的token_id可能不同
            "target_sid": s["target_sid"],  # 会在compute时用tok_cot重新算
        })
    # 简化：由于两个tokenizer的SID token id应该完全一致（都是extend后的顺序），直接用同样的target_sid即可
    # 但为保险起见，先验证：
    sample_iid = list(item_index.keys())[0]
    sids = item_index[sample_iid]
    ids_b = tok_baseline.convert_tokens_to_ids(sids)
    ids_c = tok_cot.convert_tokens_to_ids(sids)
    if ids_b != ids_c:
        print(f"[WARN] tokenizer ids differ! Rebuilding samples for CoT tokenizer")
        # 重新构造samples
        # ...
        raise RuntimeError("Tokenizer ids differ, need remapping")

    model_c, _ = load_model(args.cot_ckpt, args.tokenizer_cot, device)
    ranks_cot = compute_gt_ranks(
        model_c, tok_cot, samples, prefix_tree, item_sid_ids,
        num_beams=args.num_beams, batch_size=args.batch_size, device=device,
    )
    del model_c
    torch.cuda.empty_cache()
    print(f"耗时: {time.time()-t0:.1f}s")

    # ========== 分析 ==========
    print("\n===== 分析 =====")
    bucket_baseline = bucket_ranks(ranks_baseline)
    bucket_cot = bucket_ranks(ranks_cot)

    print(f"\n{'Bucket':<15} {'Baseline':<12} {'CoT':<12} {'Diff':<12}")
    print("-" * 55)
    for k in bucket_baseline.keys():
        b = bucket_baseline[k]
        c = bucket_cot[k]
        diff = c - b
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{k:<15} {b:<12} {c:<12} {diff:+d} {arrow}")

    # 关键分析：找出"CoT把GT从中间rank推走"的样本
    both_hit = [(rb, rc) for rb, rc in zip(ranks_baseline, ranks_cot) if rb <= args.num_beams and rc <= args.num_beams]
    baseline_hit_cot_miss = [(rb, rc) for rb, rc in zip(ranks_baseline, ranks_cot) if rb <= args.num_beams and rc > args.num_beams]
    cot_hit_baseline_miss = [(rb, rc) for rb, rc in zip(ranks_baseline, ranks_cot) if rb > args.num_beams and rc <= args.num_beams]

    print(f"\n===== Transition Analysis =====")
    print(f"Both hit@{args.num_beams}: {len(both_hit)}")
    print(f"  平均 rank: baseline={np.mean([x[0] for x in both_hit]):.2f}, cot={np.mean([x[1] for x in both_hit]):.2f}")
    print(f"Baseline hit, CoT MISS: {len(baseline_hit_cot_miss)}  ← CoT把这些推出候选集")
    if baseline_hit_cot_miss:
        print(f"  它们在baseline的平均rank: {np.mean([x[0] for x in baseline_hit_cot_miss]):.2f}")
    print(f"CoT hit, Baseline MISS: {len(cot_hit_baseline_miss)}  ← CoT救回了这些")
    if cot_hit_baseline_miss:
        print(f"  它们在CoT的平均rank: {np.mean([x[1] for x in cot_hit_baseline_miss]):.2f}")

    # 保存
    results["ranks_baseline"] = ranks_baseline
    results["ranks_cot"] = ranks_cot
    results["bucket_baseline"] = bucket_baseline
    results["bucket_cot"] = bucket_cot
    results["transition"] = {
        "both_hit": len(both_hit),
        "baseline_hit_cot_miss": len(baseline_hit_cot_miss),
        "cot_hit_baseline_miss": len(cot_hit_baseline_miss),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存到: {out}")


if __name__ == "__main__":
    main()
