"""SFT 个性化能力诊断脚本。

回答的问题：
1. Item 覆盖率：Top-K 中覆盖了多少个不同商品？
2. Head item bias：Top-1/Top-5 中最热门商品占比多少？
3. 用户间输出相似度：不同用户的 Top-5 有多相似（Jaccard）？
4. 历史敏感性：shuffle 用户历史后 Top-5 是否变化？
5. 热门 baseline：仅推荐训练集最热门 K 个商品的 HR@K 是多少？

用法：
    python scripts/analysis/diagnose_personalization.py \\
        --model_path data/checkpoints/sft/merged \\
        --tokenizer_path data/checkpoints/sft/tokenizer \\
        --sample_size 500
"""
import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加 src 到 PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from inference.beam_search import generate_topk_sids
from inference.prefix_tree import SidPrefixTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_prompt(history, item_sid_ids, tokenizer):
    """与 evaluator._build_prompt 保持一致。"""
    prefix_ids = tokenizer.encode("用户历史：", add_special_tokens=False)
    history_ids = []
    for iid in history:
        if iid in item_sid_ids:
            history_ids.extend(item_sid_ids[iid])
    suffix_ids = tokenizer.encode(" 请推荐下一个商品：", add_special_tokens=False)
    return prefix_ids + history_ids + suffix_ids


def sid_to_item_id(sid_tuple, sid_to_item):
    """SID tuple → item_id（若碰撞取任意一个，若无对应返回 None）。"""
    return sid_to_item.get(sid_tuple, None)


def compute_popularity(user_sequences, item_sid_ids):
    """基于用户历史统计商品出现次数，返回 Counter[item_id]。"""
    counter = Counter()
    for seq in user_sequences.values():
        for iid in seq:
            if iid in item_sid_ids:
                counter[iid] += 1
    return counter


def load_data(args, tokenizer):
    """加载 item_index / user_sequences / test.csv。返回诊断所需数据。"""
    logger.info(f"加载 item.index.json: {args.item_index_path}")
    with open(args.item_index_path, "r", encoding="utf-8") as f:
        item_index = json.load(f)

    item_sid_ids = {}      # item_id -> [token_id, ...]
    sid_to_item = {}       # tuple(token_ids) -> item_id  (碰撞时后覆盖前)
    for item_id_str, sid_tokens in item_index.items():
        token_ids = tokenizer.convert_tokens_to_ids(sid_tokens)
        item_id = int(item_id_str)
        item_sid_ids[item_id] = token_ids
        sid_tuple = tuple(token_ids)
        sid_to_item.setdefault(sid_tuple, []).append(item_id)
    # 保留所有碰撞的 item 列表（后面统计需要）
    sid_to_items = sid_to_item

    logger.info(f"加载 user_sequences: {args.user_sequences_path}")
    with open(args.user_sequences_path, "r", encoding="utf-8") as f:
        user_sequences = json.load(f)
    user_sequences = {int(k): v for k, v in user_sequences.items()}

    logger.info(f"加载 test.csv: {args.test_csv_path}")
    samples = []
    with open(args.test_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = int(row["user_id"])
            target_item = int(row["item_id"])
            if user_id not in user_sequences or len(user_sequences[user_id]) == 0:
                continue
            if target_item not in item_sid_ids:
                continue
            history = user_sequences[user_id][-args.max_history_len:]
            samples.append({
                "user_id": user_id,
                "history": history,
                "target_item": target_item,
                "target_sid": tuple(item_sid_ids[target_item]),
            })

    logger.info(f"测试样本总数: {len(samples)}")
    random.seed(args.random_seed)
    if args.sample_size < len(samples):
        samples = random.sample(samples, args.sample_size)
        logger.info(f"采样后样本数: {len(samples)}")

    return item_sid_ids, sid_to_items, user_sequences, samples


def run_inference(model, tokenizer, samples, item_sid_ids, prefix_tree, args, device, tag=""):
    """跑一遍推理，返回 [(target_sid, [topk_sids]), ...]"""
    results = []
    num_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    for bi in range(num_batches):
        batch = samples[bi * args.batch_size : (bi + 1) * args.batch_size]
        prompts = [build_prompt(s["history"], item_sid_ids, tokenizer) for s in batch]
        topk = generate_topk_sids(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            prefix_tree=prefix_tree,
            num_beams=args.num_beams,
            num_sid_layers=args.num_sid_layers,
            device=device,
        )
        for i, s in enumerate(batch):
            results.append({
                "user_id": s["user_id"],
                "target_sid": s["target_sid"],
                "target_item": s["target_item"],
                "topk": topk[i],
            })
        if (bi + 1) % 20 == 0 or bi + 1 == num_batches:
            logger.info(f"[{tag}] batch {bi+1}/{num_batches} 完成")
    return results


# ═══════════════ 诊断分析函数 ═══════════════

def analyze_coverage(results, sid_to_items, total_items, K_list=(1, 5, 10, 50)):
    """Item 覆盖率：所有用户 Top-K 加起来覆盖多少个不同商品。"""
    report = {}
    for K in K_list:
        all_sids = set()
        for r in results:
            for sid in r["topk"][:K]:
                all_sids.add(sid)
        # 每个 sid 可能对应多个 item（碰撞），我们统计 SID 覆盖率
        report[f"unique_sids@{K}"] = len(all_sids)
        report[f"coverage_rate@{K}"] = round(len(all_sids) / total_items * 100, 2)  # %
    return report


def analyze_head_bias(results, sid_to_items, K_list=(1, 5)):
    """Head item bias：Top-K 位置最热门商品占比。"""
    report = {}
    for K in K_list:
        counter = Counter()
        for r in results:
            for sid in r["topk"][:K]:
                counter[sid] += 1
        total = sum(counter.values())
        top10 = counter.most_common(10)
        report[f"top10_share@{K}"] = round(sum(c for _, c in top10) / total * 100, 2)  # %
        report[f"top1_share@{K}"] = round(top10[0][1] / total * 100, 2) if top10 else 0
        # 记录 Top-1 位置最集中的 3 个商品
        if K == 1:
            top3_items = []
            for sid, cnt in counter.most_common(3):
                items = sid_to_items.get(sid, [])
                item_id = items[0] if items else -1
                top3_items.append({
                    "item_id": item_id,
                    "sid": list(sid),
                    "count": cnt,
                    "share_pct": round(cnt / total * 100, 2),
                    "collision_size": len(items),
                })
            report["top1_most_predicted"] = top3_items
    return report


def analyze_pairwise_similarity(results, K=5, num_pairs=1000):
    """随机抽样用户对，计算 Top-K Jaccard 相似度分布。"""
    random.seed(0)
    n = len(results)
    if n < 2:
        return {}
    pairs = []
    tries = 0
    while len(pairs) < num_pairs and tries < num_pairs * 10:
        i, j = random.sample(range(n), 2)
        pairs.append((i, j))
        tries += 1

    jaccards = []
    exact_match = 0  # Top-K 完全相同的对数
    for i, j in pairs:
        s1 = set(results[i]["topk"][:K])
        s2 = set(results[j]["topk"][:K])
        inter = len(s1 & s2)
        union = len(s1 | s2)
        j_score = inter / union if union > 0 else 0
        jaccards.append(j_score)
        if inter == K:
            exact_match += 1

    jaccards.sort()
    n_j = len(jaccards)
    return {
        f"jaccard@{K}_mean": round(sum(jaccards) / n_j, 4),
        f"jaccard@{K}_median": round(jaccards[n_j // 2], 4),
        f"jaccard@{K}_p90": round(jaccards[int(n_j * 0.9)], 4),
        f"jaccard@{K}_exact_match_pct": round(exact_match / n_j * 100, 2),
        f"jaccard@{K}_pairs_sampled": n_j,
    }


def analyze_history_sensitivity(model, tokenizer, samples, item_sid_ids, prefix_tree,
                                 args, device, num_users=50):
    """历史敏感性：shuffle 历史 vs 原始，Top-5 是否变化。"""
    random.seed(1)
    subset = random.sample(samples, min(num_users, len(samples)))

    # 原始 Top-5
    results_orig = run_inference(model, tokenizer, subset, item_sid_ids, prefix_tree,
                                  args, device, tag="orig")

    # Shuffle 历史后再推理
    shuffled_samples = []
    for s in subset:
        h = s["history"][:]
        rng = random.Random(s["user_id"])
        rng.shuffle(h)
        shuffled_samples.append({**s, "history": h})
    results_shuf = run_inference(model, tokenizer, shuffled_samples, item_sid_ids, prefix_tree,
                                  args, device, tag="shuf")

    # Reversed 历史（更强的扰动：把最近的和最远的位置反过来）
    reversed_samples = [{**s, "history": s["history"][::-1]} for s in subset]
    results_rev = run_inference(model, tokenizer, reversed_samples, item_sid_ids, prefix_tree,
                                 args, device, tag="rev")

    def similarity(a, b, K=5):
        vals = []
        for ra, rb in zip(a, b):
            s1 = set(ra["topk"][:K])
            s2 = set(rb["topk"][:K])
            inter = len(s1 & s2)
            union = len(s1 | s2)
            vals.append(inter / union if union > 0 else 0)
        return round(sum(vals) / len(vals), 4)

    return {
        "orig_vs_shuffled_jaccard@5": similarity(results_orig, results_shuf, K=5),
        "orig_vs_reversed_jaccard@5": similarity(results_orig, results_rev, K=5),
        "orig_vs_shuffled_jaccard@1": similarity(results_orig, results_shuf, K=1),
        "num_users_probed": len(subset),
        "note": "1.0 表示完全不敏感（历史顺序无影响），越低表示越个性化",
    }


def analyze_popularity_baseline(user_sequences, item_sid_ids, samples, K_list=(1, 5, 10, 50)):
    """热门 baseline：仅推荐最热门 K 个商品的 HR@K。"""
    pop = compute_popularity(user_sequences, item_sid_ids)
    top_items = [iid for iid, _ in pop.most_common(max(K_list))]

    report = {}
    for K in K_list:
        topK_set = set(top_items[:K])
        hits = sum(1 for s in samples if s["target_item"] in topK_set)
        report[f"popularity_HR@{K}"] = round(hits / len(samples) * 100, 4)  # %
    report["popularity_top1_item"] = top_items[0]
    return report


# ═══════════════ Main ═══════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="data/checkpoints/sft/merged")
    parser.add_argument("--tokenizer_path", default="data/checkpoints/sft/tokenizer")
    parser.add_argument("--item_index_path",
                        default="data/processed/beauty_2021_2023/item.index.json")
    parser.add_argument("--user_sequences_path",
                        default="data/processed/beauty_2021_2023/user_sequences.json")
    parser.add_argument("--test_csv_path",
                        default="data/processed/beauty_2021_2023/test.csv")
    parser.add_argument("--output_path",
                        default="data/eval_results/diagnose_sft.json")
    parser.add_argument("--sample_size", type=int, default=500,
                        help="主分析样本数（越大越准，但耗时越长）")
    parser.add_argument("--history_probe_users", type=int, default=50,
                        help="历史敏感性探测用户数")
    parser.add_argument("--max_history_len", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--num_sid_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()

    # ── 加载 tokenizer + model ─────────────────────────────────────
    logger.info(f"加载 tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"加载模型: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()
    model = model.to(device)
    logger.info(f"模型加载完毕，设备: {device}")

    # ── 加载数据 ─────────────────────────────────────────────────
    item_sid_ids, sid_to_items, user_sequences, samples = load_data(args, tokenizer)
    prefix_tree = SidPrefixTree(item_sid_ids)
    logger.info(f"前缀树构建完毕: {len(item_sid_ids)} 商品, "
                f"{len(set(tuple(v) for v in item_sid_ids.values()))} 唯一 SID")

    # ── 主推理 ──────────────────────────────────────────────────
    logger.info(f"[主推理] 开始，样本数 {len(samples)}")
    t0 = time.time()
    results = run_inference(model, tokenizer, samples, item_sid_ids, prefix_tree,
                             args, device, tag="main")
    logger.info(f"[主推理] 完成，耗时 {time.time()-t0:.1f}s")

    # ── 诊断分析 ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("开始诊断分析...")

    total_items = len(item_sid_ids)
    coverage = analyze_coverage(results, sid_to_items, total_items)
    head_bias = analyze_head_bias(results, sid_to_items)
    pairwise = analyze_pairwise_similarity(results, K=5, num_pairs=2000)
    popularity = analyze_popularity_baseline(user_sequences, item_sid_ids, samples)

    logger.info("[历史敏感性] 探测中（会额外跑 2× 推理）...")
    t1 = time.time()
    history_sens = analyze_history_sensitivity(
        model, tokenizer, samples, item_sid_ids, prefix_tree,
        args, device, num_users=args.history_probe_users,
    )
    logger.info(f"[历史敏感性] 完成，耗时 {time.time()-t1:.1f}s")

    elapsed = time.time() - start

    # ── 输出 ────────────────────────────────────────────────────
    report = {
        "config": {
            "model_path": args.model_path,
            "sample_size": len(results),
            "num_beams": args.num_beams,
            "total_items": total_items,
        },
        "coverage": coverage,
        "head_bias": head_bias,
        "pairwise_similarity": pairwise,
        "history_sensitivity": history_sens,
        "popularity_baseline": popularity,
        "elapsed_seconds": round(elapsed, 2),
    }

    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 70)
    print(f"诊断报告已保存: {args.output_path}")
    print("=" * 70)
    print(f"耗时: {elapsed:.1f}s\n")

    print("【1. Item 覆盖率】(样本={}, 总商品={})".format(len(results), total_items))
    for k, v in coverage.items():
        print(f"  {k:30s}: {v}")
    print()

    print("【2. Head Item Bias】")
    for k, v in head_bias.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    - item_id={item['item_id']}, count={item['count']}, "
                      f"share={item['share_pct']}%, sid={item['sid']}, "
                      f"collision_size={item['collision_size']}")
        else:
            print(f"  {k:30s}: {v}")
    print()

    print("【3. 用户间输出相似度】")
    for k, v in pairwise.items():
        print(f"  {k:35s}: {v}")
    print()

    print("【4. 历史敏感性】")
    for k, v in history_sens.items():
        print(f"  {k:35s}: {v}")
    print()

    print("【5. 热门 Baseline HR@K (%)】")
    for k, v in popularity.items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
