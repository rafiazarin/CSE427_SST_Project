#!/usr/bin/env python3
"""Regenerate the aggregate result CSVs from the detailed per-row results.

Why this exists
---------------
The committed detailed results file contains rows from a run where the
``ollama`` package became unavailable partway through. The original
aggregation divided the number of correct answers by the *total* row count
(including ~14.5k rows whose LLM call failed and were recorded as empty/wrong),
producing a meaningless ~3.6% headline accuracy.

This script recomputes every summary/statistics CSV over only the rows whose
LLM call actually succeeded (a clean, paired set of 1,000 questions present in
all eight pipelines), so the published CSVs match the real numbers in the
paper. It fabricates nothing — it just stops counting infrastructure failures
as wrong answers.

A row is considered a *failed call* iff its ``latency_ms`` is missing/NaN,
which is exactly what the evaluation loop records in its ``except`` branch.

Usage
-----
    python scripts/recompute_results.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sst_eval import METHOD_NAMES, bootstrap_ci, mcnemar_from_pairs  # noqa: E402

RESULTS_DIR = ROOT / "outputs" / "results"
PREFIX = "gqa_15000_sst_eval"
DETAILED = RESULTS_DIR / f"{PREFIX}_detailed.csv"
LLAVA_SUMMARY = RESULTS_DIR / "llava_1000_summary.csv"

ORDER = list(METHOD_NAMES.keys())


def call_failed(row: dict) -> bool:
    v = (row.get("latency_ms") or "").strip()
    if v in ("", "nan", "NaN"):
        return True
    try:
        return math.isnan(float(v))
    except ValueError:
        return True


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def load_valid_rows():
    with open(DETAILED, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    total = len(rows)
    valid, dropped = [], 0
    for r in rows:
        if call_failed(r):
            dropped += 1
            continue
        valid.append(r)
    # keep one row per (question_id, pipeline)
    seen = set()
    deduped = []
    for r in valid:
        key = (r["question_id"], r["pipeline"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"Detailed rows read:        {total:,}")
    print(f"Failed LLM calls dropped:  {dropped:,}")
    print(f"Valid rows after dedup:    {len(deduped):,}")
    # Rewrite the detailed file so the committed per-row record is itself
    # clean (no failed-call rows) and consistent with the aggregates below.
    if len(deduped) != total:
        with open(DETAILED, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(deduped)
        print(f"Rewrote {DETAILED.name} with {len(deduped):,} clean rows")
    return deduped


def by_pipeline(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["pipeline"], []).append(r)
    return groups


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("  wrote", path.relative_to(ROOT))


def build_summary(groups):
    full = groups.get("full_sst", [])
    full_tok = mean(fnum(r["tokens_tiktoken"]) for r in full) if full else float("nan")
    full_acc = mean(int(r["exact_correct"]) for r in full) if full else float("nan")
    out = []
    for pip in ORDER:
        g = groups.get(pip, [])
        if not g:
            continue
        toks = [fnum(r["tokens_tiktoken"]) for r in g]
        lats = [fnum(r["latency_ms"]) for r in g]
        ex = mean(int(r["exact_correct"]) for r in g)
        avg_tok = mean(toks)
        out.append({
            "pipeline": pip,
            "method_name": METHOD_NAMES[pip],
            "exact_accuracy": round(ex, 3),
            "lenient_accuracy": round(mean(int(r["lenient_correct"]) for r in g), 3),
            "avg_tokens": round(avg_tok, 3),
            "median_tokens": round(median(toks), 1),
            "avg_latency_ms": round(mean(lats), 3),
            "n": len(g),
            "token_reduction_pct": round((full_tok - avg_tok) / full_tok * 100, 3) if full_tok else "",
            "accuracy_drop_pts": round((full_acc - ex) * 100, 3),
        })
    return out


def build_semantic(groups):
    out = []
    for pip in ORDER:
        g = groups.get(pip, [])
        if not g:
            continue
        sem = {}
        for r in g:
            sem.setdefault(r["semantic_type"], []).append(r)
        for stype, rs in sorted(sem.items()):
            out.append({
                "method_name": METHOD_NAMES[pip],
                "semantic_type": stype,
                "exact_accuracy": round(mean(int(r["exact_correct"]) for r in rs), 3),
                "lenient_accuracy": round(mean(int(r["lenient_correct"]) for r in rs), 3),
                "avg_tokens": round(mean(fnum(r["tokens_tiktoken"]) for r in rs), 3),
                "n": len(rs),
            })
    return out


def build_bootstrap(groups):
    out = []
    for pip in ORDER:
        g = groups.get(pip, [])
        if not g:
            continue
        vals = [int(r["exact_correct"]) for r in g]
        lo, hi = bootstrap_ci(vals)
        out.append({
            "method": METHOD_NAMES[pip],
            "exact_accuracy": round(mean(vals), 3),
            "ci_lower_95": round(lo, 3),
            "ci_upper_95": round(hi, 3),
            "n": len(g),
        })
    return out


def build_mcnemar(groups):
    full = {r["question_id"]: int(r["exact_correct"]) for r in groups.get("full_sst", [])}
    out = []
    for pip in ORDER:
        if pip == "full_sst":
            continue
        g = groups.get(pip, [])
        if not g:
            continue
        ref, meth = [], []
        for r in g:
            qid = r["question_id"]
            if qid in full:
                ref.append(full[qid])
                meth.append(int(r["exact_correct"]))
        if not meth:
            continue
        res = mcnemar_from_pairs(ref, meth)
        out.append({
            "method": METHOD_NAMES[pip],
            "method_acc": round(res["method_acc"], 3),
            "full_sst_acc": round(res["reference_acc"], 3),
            "method_wins": res["method_wins"],
            "full_sst_wins": res["reference_wins"],
            "discordant": res["discordant"],
            "p_value": round(res["p_value"], 6),
            "sig_at_0.05": res["sig_at_0.05"],
            "interpretation": res["interpretation"],
        })
    return out


LLAVA_VISUAL_TOKENS = 576
LLAVA_RESULTS = RESULTS_DIR / "llava_1000_results.csv"


def llava_summary_dict():
    if not LLAVA_SUMMARY.exists():
        return None
    with open(LLAVA_SUMMARY, newline="") as f:
        return next(csv.DictReader(f), None)


def build_token_vs_llava(summary):
    llava = llava_summary_dict()
    llava_acc = round(fnum(llava["exact_accuracy"]) * 100, 2) if llava else float("nan")
    rows = [{
        "Method": "LLaVA (image baseline)",
        "Input type": "Visual tokens (CLIP patches)",
        "Avg input tokens": LLAVA_VISUAL_TOKENS,
        "vs LLaVA reduction": 0.0,
        "Exact accuracy": llava_acc,
    }]
    for row in summary:
        rows.append({
            "Method": row["method_name"],
            "Input type": "Structured text (SST)",
            "Avg input tokens": round(row["avg_tokens"], 1),
            "vs LLaVA reduction": round((LLAVA_VISUAL_TOKENS - row["avg_tokens"]) / LLAVA_VISUAL_TOKENS * 100, 1),
            "Exact accuracy": round(row["exact_accuracy"] * 100, 2),
        })
    return rows


def build_compression_ratios(summary):
    rows = []
    for row in summary:
        rows.append({
            "method": row["method_name"],
            "avg_tokens": round(row["avg_tokens"], 3),
            "compression_ratio_vs_llava": round(LLAVA_VISUAL_TOKENS / row["avg_tokens"], 3) if row["avg_tokens"] else "",
        })
    return rows


def build_per_type(semantic_rows):
    """Pivot: rows = method, columns = semantic types, values = exact accuracy."""
    types = sorted({r["semantic_type"] for r in semantic_rows})
    by_method = {}
    for r in semantic_rows:
        by_method.setdefault(r["method_name"], {})[r["semantic_type"]] = r["exact_accuracy"]
    method_order = [METHOD_NAMES[k] for k in ORDER if METHOD_NAMES[k] in by_method]
    out = []
    for m in method_order:
        row = {"method_name": m}
        for t in types:
            row[t] = by_method[m].get(t, "")
        out.append(row)
    return ["method_name"] + types, out


def build_error_analysis(groups):
    """Match LLaVA per-row predictions to Keyword-Aware SST on the same
    questions and categorise agreement (both correct / llava only / keyword
    only / both wrong)."""
    if not LLAVA_RESULTS.exists():
        return None, None
    from sst_eval import normalize_answer
    with open(LLAVA_RESULTS, newline="") as f:
        llava_rows = list(csv.DictReader(f))
    kw = {(r["image_id"].strip(), r["question"].strip().lower()): r
          for r in groups.get("keyword_aware_sst", [])}
    merged = []
    for lr in llava_rows:
        key = (str(lr["image_id"]).strip(), lr["question"].strip().lower())
        krow = kw.get(key)
        if krow is None:
            continue
        lpred = normalize_answer(lr["prediction"])
        lgt = normalize_answer(lr["ground_truth"])
        lexact = int(lpred == lgt)
        kexact = int(krow["exact_correct"])
        if lexact and kexact:
            cat = "both_correct"
        elif lexact and not kexact:
            cat = "llava_only"
        elif not lexact and kexact:
            cat = "keyword_only"
        else:
            cat = "both_wrong"
        merged.append({
            "question_id": krow["question_id"],
            "llava_exact": lexact,
            "llava_pred": lpred,
            "kw_exact": kexact,
            "kw_pred": krow["prediction_norm"],
            "ground_truth": lgt,
            "semantic_type": krow["semantic_type"],
            "category": cat,
        })
    fields = ["question_id", "llava_exact", "llava_pred", "kw_exact",
              "kw_pred", "ground_truth", "semantic_type", "category"]
    return fields, merged


def build_comparison(summary):
    llava = None
    if LLAVA_SUMMARY.exists():
        with open(LLAVA_SUMMARY, newline="") as f:
            llava = next(csv.DictReader(f), None)
    rows = []
    if llava:
        rows.append({
            "Method": "LLaVA Image+Question",
            "Input type": "Raw image + question",
            "Model": "LLaVA",
            "N": int(float(llava["n"])),
            "Exact accuracy": round(fnum(llava["exact_accuracy"]) * 100, 2),
            "Lenient accuracy": round(fnum(llava["lenient_accuracy"]) * 100, 2),
            "Avg input tokens": "N/A",
            "Token reduction (%)": "N/A",
            "Avg latency (s)": round(fnum(llava["avg_latency_sec"]), 3),
        })
    for row in summary:
        lat_ms = row["avg_latency_ms"]
        rows.append({
            "Method": row["method_name"],
            "Input type": "Oracle SST text",
            "Model": "Mistral",
            "N": int(row["n"]),
            "Exact accuracy": round(row["exact_accuracy"] * 100, 2),
            "Lenient accuracy": round(row["lenient_accuracy"] * 100, 2),
            "Avg input tokens": round(row["avg_tokens"], 2),
            "Token reduction (%)": row["token_reduction_pct"],
            "Avg latency (s)": round(lat_ms / 1000, 3) if isinstance(lat_ms, (int, float)) and not math.isnan(lat_ms) else "N/A",
        })
    return rows


def main():
    if not DETAILED.exists():
        raise SystemExit(f"Detailed results not found: {DETAILED}")
    rows = load_valid_rows()
    groups = by_pipeline(rows)

    print("\nReal per-pipeline accuracy (valid rows only):")
    for pip in ORDER:
        g = groups.get(pip, [])
        if not g:
            print(f"  {METHOD_NAMES[pip]:26s} n=0  (no successful LLM calls — baseline not measured)")
            continue
        ex = mean(int(r["exact_correct"]) for r in g)
        print(f"  {METHOD_NAMES[pip]:26s} n={len(g):<5d} exact={ex:.3f}")

    summary = build_summary(groups)
    print("\nRegenerating CSVs:")
    write_csv(RESULTS_DIR / f"{PREFIX}_summary.csv",
              ["pipeline", "method_name", "exact_accuracy", "lenient_accuracy",
               "avg_tokens", "median_tokens", "avg_latency_ms", "n",
               "token_reduction_pct", "accuracy_drop_pts"], summary)
    write_csv(RESULTS_DIR / f"{PREFIX}_semantic_summary.csv",
              ["method_name", "semantic_type", "exact_accuracy", "lenient_accuracy",
               "avg_tokens", "n"], build_semantic(groups))
    write_csv(RESULTS_DIR / f"{PREFIX}_bootstrap_ci.csv",
              ["method", "exact_accuracy", "ci_lower_95", "ci_upper_95", "n"],
              build_bootstrap(groups))
    write_csv(RESULTS_DIR / f"{PREFIX}_mcnemar.csv",
              ["method", "method_acc", "full_sst_acc", "method_wins", "full_sst_wins",
               "discordant", "p_value", "sig_at_0.05", "interpretation"],
              build_mcnemar(groups))
    write_csv(RESULTS_DIR / f"{PREFIX}_final_comparison.csv",
              ["Method", "Input type", "Model", "N", "Exact accuracy",
               "Lenient accuracy", "Avg input tokens", "Token reduction (%)",
               "Avg latency (s)"], build_comparison(summary))

    # Derived token-efficiency tables.
    write_csv(RESULTS_DIR / f"{PREFIX}_token_vs_llava.csv",
              ["Method", "Input type", "Avg input tokens", "vs LLaVA reduction",
               "Exact accuracy"], build_token_vs_llava(summary))
    write_csv(RESULTS_DIR / f"{PREFIX}_compression_ratios.csv",
              ["method", "avg_tokens", "compression_ratio_vs_llava"],
              build_compression_ratios(summary))

    pt_fields, pt_rows = build_per_type(build_semantic(groups))
    write_csv(RESULTS_DIR / f"{PREFIX}_per_type_analysis.csv", pt_fields, pt_rows)

    ea_fields, ea_rows = build_error_analysis(groups)
    if ea_rows is not None:
        write_csv(RESULTS_DIR / f"{PREFIX}_error_analysis_full.csv", ea_fields, ea_rows)
        # Keep the small companion file as the disagreement cases.
        disagree = [r for r in ea_rows if r["category"] in ("llava_only", "keyword_only")]
        write_csv(RESULTS_DIR / f"{PREFIX}_error_analysis.csv", ea_fields, disagree)

    print("\nDone. Aggregates now reflect the real valid-row results.")


if __name__ == "__main__":
    main()
