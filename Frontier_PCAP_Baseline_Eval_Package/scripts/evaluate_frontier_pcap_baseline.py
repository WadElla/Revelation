#!/usr/bin/env python3
"""
Final checked ChatGPT PCAP-only answer-quality evaluation for the Revelation frontier baseline.

This script compares saved ChatGPT PCAP-only answers against the gold/reference QA answers.
It does not run Revelation, RAG, Chroma, Zeek, BERT, PCAP processing, or answer generation.

Primary evaluation:
    all reference questions are included.
    explicit ChatGPT refusals are scored against the reference answers.
    blank/missing predictions are scored as 0.0, not ignored.

Secondary evaluation:
    answered-only summaries exclude explicit refusal answers and blank/missing predictions.

Metrics:
    BLEU
    ROUGE-1 F1
    ROUGE-2 F1
    ROUGE-L F1
    official NLTK METEOR
    BERTScore RAW, implemented to match the provided reference code:
        bert_score([prediction], [reference], lang="en", verbose=False)
      but computed in batch for efficiency.
    BERTScore STRICT:
        roberta-large, IDF=True, rescale_with_baseline=True
    normalized exact match

Expected CSV columns:
    reference CSV:  scenario, question_number, question, reference_answer
    prediction CSV: scenario, question_number, question, predicted_answer

Optional columns preserved when present:
    question_category, model_name, source_docx
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except Exception as exc:  # pragma: no cover
    nltk = None
    SmoothingFunction = None
    sentence_bleu = None
    meteor_score = None
    _NLTK_IMPORT_ERROR = repr(exc)
else:
    _NLTK_IMPORT_ERROR = ""

try:
    from rouge_score import rouge_scorer
except Exception as exc:  # pragma: no cover
    rouge_scorer = None
    _ROUGE_IMPORT_ERROR = repr(exc)
else:
    _ROUGE_IMPORT_ERROR = ""

try:
    from bert_score import score as bert_score
except Exception as exc:  # pragma: no cover
    bert_score = None
    _BERTSCORE_IMPORT_ERROR = repr(exc)
else:
    _BERTSCORE_IMPORT_ERROR = ""


BASE_METRIC_COLUMNS = [
    "bleu",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "meteor",
    "exact_match_norm",
]

BERTSCORE_COLUMNS = [
    "bertscore_raw_precision",
    "bertscore_raw_recall",
    "bertscore_raw_f1",
    "bertscore_strict_precision",
    "bertscore_strict_recall",
    "bertscore_strict_f1",
]

ALL_METRIC_COLUMNS = BASE_METRIC_COLUMNS + BERTSCORE_COLUMNS

REFUSAL_PATTERNS = [
    r"cannot\s+be\s+determined",
    r"cannot\s+answer",
    r"not\s+determinable",
    r"unable\s+to\s+determine",
    r"insufficient\s+evidence",
    r"not\s+enough\s+evidence",
    r"no\s+sufficient\s+(pcap\s+)?evidence",
    r"not\s+present\s+in\s+the\s+pcap",
    r"not\s+available\s+from\s+the\s+pcap",
    r"cannot\s+be\s+verified\s+from\s+the\s+pcap",
    r"not\s+provided\s+in\s+the\s+pcap",
    r"not\s+visible\s+in\s+the\s+pcap",
]


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------
def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\\u2192", "→")
    text = text.replace("\u00a0", " ").replace("\xa0", " ")
    text = text.replace("`", "")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_exact(text: Any) -> str:
    text = normalize_text(text).lower()
    text = text.replace("->", "→")
    text = re.sub(r"[^a-z0-9_.:/\-→ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_tokens(text: Any) -> List[str]:
    """Whitespace tokenization to match the supplied reference evaluation code."""
    return normalize_text(text).split()


def is_refusal(text: Any) -> bool:
    t = normalize_text(text).lower()
    return any(re.search(pattern, t) for pattern in REFUSAL_PATTERNS)


def safe_mean(values: Iterable[Any]) -> Optional[float]:
    nums: List[float] = []
    for v in values:
        try:
            if pd.isna(v):
                continue
            nums.append(float(v))
        except Exception:
            continue
    return sum(nums) / len(nums) if nums else None


def format_float(v: Any) -> str:
    try:
        if v is None or pd.isna(v):
            return "n/a"
        return f"{float(v):.4f}"
    except Exception:
        return "n/a"


# -----------------------------------------------------------------------------
# Dependency checks
# -----------------------------------------------------------------------------
def check_metric_dependencies(use_bertscore: bool) -> None:
    missing: List[str] = []
    if nltk is None or sentence_bleu is None or SmoothingFunction is None or meteor_score is None:
        missing.append(f"nltk with BLEU/METEOR support ({_NLTK_IMPORT_ERROR})")
    if rouge_scorer is None:
        missing.append(f"rouge-score ({_ROUGE_IMPORT_ERROR})")
    if use_bertscore and bert_score is None:
        missing.append(f"bert-score ({_BERTSCORE_IMPORT_ERROR})")
    if missing:
        raise RuntimeError(
            "Required metric dependencies are missing:\n- "
            + "\n- ".join(missing)
            + "\nInstall the requirements before running the final evaluation."
        )


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path).fillna("")


def validate_inputs(ref_df: pd.DataFrame, pred_df: pd.DataFrame, strict: bool = True) -> Dict[str, Any]:
    required_ref = {"scenario", "question_number", "question", "reference_answer"}
    required_pred = {"scenario", "question_number", "question", "predicted_answer"}

    missing_ref_cols = sorted(required_ref - set(ref_df.columns))
    missing_pred_cols = sorted(required_pred - set(pred_df.columns))
    if missing_ref_cols:
        raise ValueError(f"Reference CSV missing columns: {missing_ref_cols}")
    if missing_pred_cols:
        raise ValueError(f"Prediction CSV missing columns: {missing_pred_cols}")

    ref_tmp = ref_df.copy()
    pred_tmp = pred_df.copy()
    ref_tmp["question_number"] = ref_tmp["question_number"].astype(int)
    pred_tmp["question_number"] = pred_tmp["question_number"].astype(int)

    key_cols = ["scenario", "question_number"]
    ref_duplicate_rows = ref_tmp[ref_tmp.duplicated(key_cols, keep=False)]
    pred_duplicate_rows = pred_tmp[pred_tmp.duplicated(key_cols, keep=False)]
    ref_duplicate_keys = sorted(
        set(zip(ref_duplicate_rows["scenario"].astype(str), ref_duplicate_rows["question_number"]))
    )
    pred_duplicate_keys = sorted(
        set(zip(pred_duplicate_rows["scenario"].astype(str), pred_duplicate_rows["question_number"]))
    )

    ref_keys = set(zip(ref_tmp["scenario"].astype(str), ref_tmp["question_number"]))
    pred_keys = set(zip(pred_tmp["scenario"].astype(str), pred_tmp["question_number"]))

    missing_prediction_keys = sorted(list(ref_keys - pred_keys))
    extra_prediction_keys = sorted(list(pred_keys - ref_keys))

    q_merge = ref_tmp[["scenario", "question_number", "question"]].merge(
        pred_tmp[["scenario", "question_number", "question"]],
        on=["scenario", "question_number"],
        how="inner",
        suffixes=("_reference", "_prediction"),
    )
    q_merge["question_match_norm"] = q_merge.apply(
        lambda r: normalize_for_exact(r["question_reference"]) == normalize_for_exact(r["question_prediction"]),
        axis=1,
    )
    question_mismatch_rows = q_merge[~q_merge["question_match_norm"]]
    question_mismatches = question_mismatch_rows[
        ["scenario", "question_number", "question_reference", "question_prediction"]
    ].to_dict("records")

    report = {
        "num_reference_rows": int(len(ref_tmp)),
        "num_prediction_rows": int(len(pred_tmp)),
        "reference_counts_by_scenario": ref_tmp.groupby("scenario").size().to_dict(),
        "prediction_counts_by_scenario": pred_tmp.groupby("scenario").size().to_dict(),
        "missing_prediction_keys": missing_prediction_keys,
        "extra_prediction_keys": extra_prediction_keys,
        "duplicate_reference_keys": ref_duplicate_keys,
        "duplicate_prediction_keys": pred_duplicate_keys,
        "question_mismatch_count": int(len(question_mismatches)),
        "question_mismatches": question_mismatches,
    }

    if strict and (missing_prediction_keys or extra_prediction_keys or question_mismatches or ref_duplicate_keys or pred_duplicate_keys):
        raise ValueError("Input mismatch detected:\n" + json.dumps(report, indent=2, ensure_ascii=False))

    return report


# -----------------------------------------------------------------------------
# Official METEOR support
# -----------------------------------------------------------------------------
def ensure_official_meteor(download_nltk: bool) -> Dict[str, Any]:
    """Ensure official NLTK METEOR can run. Fail rather than silently falling back."""
    if nltk is None or meteor_score is None:
        raise RuntimeError(f"NLTK/METEOR is not available. Import error: {_NLTK_IMPORT_ERROR}")

    attempts: List[str] = []

    def _probe() -> None:
        # Force the official NLTK METEOR path to access WordNet-backed resources,
        # not only exact-token matching. This catches missing WordNet/OMW resources
        # before the real evaluation starts.
        from nltk.corpus import wordnet as wn

        if not wn.synsets("car"):
            raise RuntimeError("NLTK WordNet loaded but returned no synsets for the probe token.")
        _ = meteor_score([["car"]], ["automobile"])

    try:
        _probe()
        return {"official_meteor": True, "download_attempted": False, "notes": attempts}
    except Exception as first_exc:
        attempts.append(f"Initial METEOR probe failed: {repr(first_exc)}")

    if download_nltk:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            attempts.append("Attempted nltk.download('wordnet') and nltk.download('omw-1.4').")
        except Exception as dl_exc:
            attempts.append(f"NLTK download failed: {repr(dl_exc)}")

    try:
        _probe()
        return {"official_meteor": True, "download_attempted": bool(download_nltk), "notes": attempts}
    except Exception as final_exc:
        attempts.append(f"Final METEOR probe failed: {repr(final_exc)}")
        raise RuntimeError(
            "Official NLTK METEOR could not run. Install/download NLTK resources first, e.g.:\n"
            "python - <<'PY'\nimport nltk\nnltk.download('wordnet')\nnltk.download('omw-1.4')\nPY\n"
            + "\nDetails:\n"
            + "\n".join(attempts)
        )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_base_metrics(predicted: str, reference: str) -> Dict[str, float]:
    """Compute BLEU, ROUGE, official METEOR, and exact match for one pair."""
    predicted = normalize_text(predicted)
    reference = normalize_text(reference)
    pred_tokens = split_tokens(predicted)
    ref_tokens = split_tokens(reference)

    out: Dict[str, float] = {
        "bleu": 0.0,
        "rouge1_f1": 0.0,
        "rouge2_f1": 0.0,
        "rougeL_f1": 0.0,
        "meteor": 0.0,
        "exact_match_norm": 0.0,
    }

    if not predicted or not reference:
        return out

    out["exact_match_norm"] = 1.0 if normalize_for_exact(predicted) == normalize_for_exact(reference) else 0.0

    if sentence_bleu is None or SmoothingFunction is None:
        raise RuntimeError(f"NLTK BLEU is not available. Import error: {_NLTK_IMPORT_ERROR}")
    try:
        out["bleu"] = float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1))
    except Exception:
        out["bleu"] = 0.0

    if rouge_scorer is None:
        raise RuntimeError(f"rouge_score is not available. Import error: {_ROUGE_IMPORT_ERROR}")
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, predicted)
        out["rouge1_f1"] = float(scores["rouge1"].fmeasure)
        out["rouge2_f1"] = float(scores["rouge2"].fmeasure)
        out["rougeL_f1"] = float(scores["rougeL"].fmeasure)
    except Exception:
        out["rouge1_f1"] = 0.0
        out["rouge2_f1"] = 0.0
        out["rougeL_f1"] = 0.0

    if meteor_score is None:
        raise RuntimeError(f"NLTK METEOR is not available. Import error: {_NLTK_IMPORT_ERROR}")
    try:
        # Same tokenized calling convention as the evaluation code you provided.
        out["meteor"] = float(meteor_score([ref_tokens], pred_tokens))
    except Exception as exc:
        raise RuntimeError(f"Official METEOR failed for a QA pair: {repr(exc)}") from exc

    return out


def _assign_bertscore_values(
    df: pd.DataFrame,
    idxs: List[int],
    prefix: str,
    precision: Any,
    recall: Any,
    f1: Any,
) -> None:
    for pos, original_idx in enumerate(idxs):
        df.at[original_idx, f"{prefix}_precision"] = float(precision[pos].item())
        df.at[original_idx, f"{prefix}_recall"] = float(recall[pos].item())
        df.at[original_idx, f"{prefix}_f1"] = float(f1[pos].item())


def add_dual_bertscore(
    df: pd.DataFrame,
    raw_batch_size: int,
    strict_batch_size: int,
    raw_model_type: Optional[str],
    strict_model_type: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add raw and strict BERTScore columns. Empty predictions keep 0.0."""
    if bert_score is None:
        raise RuntimeError(f"bert_score package is not available. Import error: {_BERTSCORE_IMPORT_ERROR}")

    df = df.copy()
    for col in BERTSCORE_COLUMNS:
        df[col] = 0.0

    preds = df["predicted_answer"].fillna("").astype(str).map(normalize_text).tolist()
    refs = df["reference_answer"].fillna("").astype(str).map(normalize_text).tolist()
    non_empty_idx = [i for i, (p, r) in enumerate(zip(preds, refs)) if p and r]
    if not non_empty_idx:
        raise RuntimeError("No non-empty prediction/reference pairs available for BERTScore.")

    pred_nonempty = [preds[i] for i in non_empty_idx]
    ref_nonempty = [refs[i] for i in non_empty_idx]

    status: Dict[str, Any] = {
        "non_empty_pairs_scored": len(non_empty_idx),
        "empty_or_missing_predictions_scored_as_zero": int(len(df) - len(non_empty_idx)),
        "raw_bertscore": {
            "implementation_note": "Reference-compatible raw BERTScore: bert_score(preds, refs, lang='en', verbose=False), batched for efficiency. Explicit idf=False and rescale_with_baseline=False are included to make the defaults auditable.",
            "matches_provided_reference_code_semantics": True,
            "lang": "en",
            "idf": False,
            "rescale_with_baseline": False,
            "model_type": raw_model_type,
            "batch_size": raw_batch_size,
        },
        "strict_bertscore": {
            "implementation_note": "Diagnostic strict BERTScore with IDF weighting and baseline rescaling.",
            "lang": "en",
            "idf": True,
            "rescale_with_baseline": True,
            "model_type": strict_model_type,
            "batch_size": strict_batch_size,
        },
    }

    # RAW BERTScore: this matches the provided reference implementation except that
    # it is batched across all QA pairs instead of called once per row.
    raw_kwargs: Dict[str, Any] = {
        "lang": "en",
        "verbose": False,
        "batch_size": raw_batch_size,
        "idf": False,
        "rescale_with_baseline": False,
    }
    if raw_model_type:
        raw_kwargs["model_type"] = raw_model_type

    try:
        p, r, f1 = bert_score(pred_nonempty, ref_nonempty, **raw_kwargs)
        _assign_bertscore_values(df, non_empty_idx, "bertscore_raw", p, r, f1)
        status["raw_bertscore"]["computed"] = True
    except Exception as exc:
        status["raw_bertscore"]["computed"] = False
        status["raw_bertscore"]["error"] = repr(exc)
        raise RuntimeError(f"Raw BERTScore failed: {repr(exc)}") from exc

    # STRICT BERTScore: less forgiving diagnostic score for the PCAP-only baseline.
    strict_kwargs: Dict[str, Any] = {
        "lang": "en",
        "model_type": strict_model_type,
        "verbose": False,
        "batch_size": strict_batch_size,
        "idf": True,
        "rescale_with_baseline": True,
    }
    try:
        p, r, f1 = bert_score(pred_nonempty, ref_nonempty, **strict_kwargs)
        _assign_bertscore_values(df, non_empty_idx, "bertscore_strict", p, r, f1)
        status["strict_bertscore"]["computed"] = True
    except Exception as exc:
        status["strict_bertscore"]["computed"] = False
        status["strict_bertscore"]["error"] = repr(exc)
        raise RuntimeError(f"Strict BERTScore failed: {repr(exc)}") from exc

    return df, status


# -----------------------------------------------------------------------------
# Scoring and summaries
# -----------------------------------------------------------------------------
def build_scored_dataframe(
    ref_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_name: str,
    use_bertscore: bool,
    raw_bertscore_batch_size: int,
    strict_bertscore_batch_size: int,
    raw_bertscore_model: Optional[str],
    strict_bertscore_model: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ref_df = ref_df.copy()
    pred_df = pred_df.copy()
    ref_df["question_number"] = ref_df["question_number"].astype(int)
    pred_df["question_number"] = pred_df["question_number"].astype(int)

    pred_cols = ["scenario", "question_number", "predicted_answer"]
    for optional_col in ["model_name", "source_docx"]:
        if optional_col in pred_df.columns:
            pred_cols.append(optional_col)

    merged = ref_df.merge(pred_df[pred_cols], on=["scenario", "question_number"], how="left")

    if "model_name" not in merged.columns:
        merged["model_name"] = model_name
    merged["model_name"] = merged["model_name"].replace("", model_name).fillna(model_name)

    for col in ["question", "reference_answer", "predicted_answer"]:
        merged[col] = merged[col].fillna("").map(normalize_text)

    if "question_category" not in merged.columns:
        merged["question_category"] = "uncategorized"
    merged["question_category"] = merged["question_category"].replace("", "uncategorized").fillna("uncategorized")

    metric_rows = [
        compute_base_metrics(predicted, reference)
        for predicted, reference in zip(merged["predicted_answer"], merged["reference_answer"])
    ]
    scored = pd.concat([merged.reset_index(drop=True), pd.DataFrame(metric_rows).reset_index(drop=True)], axis=1)

    bert_status: Dict[str, Any] = {"bertscore_requested": bool(use_bertscore), "bertscore_computed": False}
    for col in BERTSCORE_COLUMNS:
        scored[col] = 0.0

    if use_bertscore:
        scored, bert_status = add_dual_bertscore(
            scored,
            raw_batch_size=raw_bertscore_batch_size,
            strict_batch_size=strict_bertscore_batch_size,
            raw_model_type=raw_bertscore_model,
            strict_model_type=strict_bertscore_model,
        )
        bert_status["bertscore_computed"] = True

    scored["chatgpt_refused"] = scored["predicted_answer"].map(is_refusal).astype(int)
    scored["prediction_missing"] = scored["predicted_answer"].eq("").astype(int)
    scored["answered_non_refusal"] = ((scored["chatgpt_refused"] == 0) & (scored["prediction_missing"] == 0)).astype(int)
    scored["reference_len_tokens"] = scored["reference_answer"].map(lambda x: len(split_tokens(x)))
    scored["prediction_len_tokens"] = scored["predicted_answer"].map(lambda x: len(split_tokens(x)))

    ranking_cols = ["bertscore_strict_f1", "rougeL_f1", "meteor"]
    scored["qualitative_review_score"] = scored[ranking_cols].astype(float).mean(axis=1)

    return scored, bert_status


def summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if group_cols:
        grouped = df.groupby(list(group_cols), dropna=False)
    else:
        grouped = [((), df)]

    records: List[Dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: val for col, val in zip(group_cols, key)}
        rec["num_questions"] = int(len(group))
        rec["chatgpt_refusal_count"] = int(group["chatgpt_refused"].sum())
        rec["chatgpt_refusal_rate"] = float(group["chatgpt_refused"].mean()) if len(group) else None
        rec["prediction_missing_count"] = int(group["prediction_missing"].sum())
        rec["answered_non_refusal_count"] = int(group["answered_non_refusal"].sum())
        rec["answered_non_refusal_rate"] = float(group["answered_non_refusal"].mean()) if len(group) else None
        rec["mean_reference_len_tokens"] = safe_mean(group["reference_len_tokens"])
        rec["mean_prediction_len_tokens"] = safe_mean(group["prediction_len_tokens"])
        for metric in ALL_METRIC_COLUMNS:
            if metric in group.columns:
                rec[f"mean_{metric}"] = safe_mean(group[metric])
        records.append(rec)

    return pd.DataFrame(records)


def write_outputs(
    scored: pd.DataFrame,
    out_dir: Path,
    validation_report: Dict[str, Any],
    meteor_status: Dict[str, Any],
    bert_status: Dict[str, Any],
    use_bertscore: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    scored.to_csv(out_dir / "per_question_scores.csv", index=False)

    # Primary: all questions, including explicit refusals and blank/missing predictions.
    all_overall = summarize(scored, ["model_name"])
    all_scenario = summarize(scored, ["model_name", "scenario"])
    all_category = summarize(scored, ["model_name", "question_category"])

    all_overall.to_csv(out_dir / "overall_metrics_all_questions.csv", index=False)
    all_scenario.to_csv(out_dir / "scenario_metrics_all_questions.csv", index=False)
    all_category.to_csv(out_dir / "category_metrics_all_questions.csv", index=False)

    # Secondary: answered-only, excluding explicit refusals and blank/missing predictions.
    answered = scored[(scored["answered_non_refusal"] == 1)].copy()
    answered_overall = summarize(answered, ["model_name"])
    answered_scenario = summarize(answered, ["model_name", "scenario"])
    answered_category = summarize(answered, ["model_name", "question_category"])

    answered_overall.to_csv(out_dir / "overall_metrics_answered_only.csv", index=False)
    answered_scenario.to_csv(out_dir / "scenario_metrics_answered_only.csv", index=False)
    answered_category.to_csv(out_dir / "category_metrics_answered_only.csv", index=False)

    scored[scored["chatgpt_refused"] == 1].to_csv(
        out_dir / "chatgpt_refusals_for_qualitative_analysis.csv",
        index=False,
    )

    scored.sort_values(
        ["qualitative_review_score", "scenario", "question_number"],
        ascending=[True, True, True],
    ).head(80).to_csv(out_dir / "low_score_questions_for_qualitative_analysis.csv", index=False)

    answered.sort_values(
        ["qualitative_review_score", "scenario", "question_number"],
        ascending=[True, True, True],
    ).head(80).to_csv(out_dir / "low_score_answered_only_questions_for_qualitative_analysis.csv", index=False)

    metric_status = {
        "evaluation_semantics": {
            "all_reference_rows_are_included_in_primary_summary_metrics": True,
            "explicit_refusal_answers_are_scored_against_reference_answers": True,
            "empty_or_missing_predictions_are_scored_as_zero": True,
            "answered_only_view_excludes_explicit_refusals_and_blank_predictions": True,
        },
        "computed_metrics": {
            "bleu": sentence_bleu is not None,
            "rouge": rouge_scorer is not None,
            "official_meteor": True,
            "bertscore_requested": bool(use_bertscore),
            "bertscore_computed": bool(bert_status.get("bertscore_computed", False)),
            "raw_bertscore_computed": bool(bert_status.get("raw_bertscore", {}).get("computed", False)),
            "strict_bertscore_computed": bool(bert_status.get("strict_bertscore", {}).get("computed", False)),
        },
        "meteor_status": meteor_status,
        "bertscore_status": bert_status,
        "input_validation": validation_report,
    }
    (out_dir / "metric_status.json").write_text(json.dumps(metric_status, indent=2, ensure_ascii=False), encoding="utf-8")

    overall = all_overall.iloc[0].to_dict() if not all_overall.empty else {}
    answered_summary = answered_overall.iloc[0].to_dict() if not answered_overall.empty else {}

    lines = [
        "ChatGPT PCAP-Only Answer Quality Evaluation",
        "============================================",
        "",
        "This run compares saved ChatGPT PCAP-only answers against the gold/reference answers.",
        "It does not run Revelation, RAG, Chroma, Zeek, BERT, PCAP processing, or answer generation.",
        "",
        "Primary all-question evaluation:",
        f"- Questions evaluated: {int(overall.get('num_questions', 0))}",
        f"- ChatGPT refusal count: {int(overall.get('chatgpt_refusal_count', 0))}",
        f"- ChatGPT refusal rate: {format_float(overall.get('chatgpt_refusal_rate'))}",
        f"- BLEU: {format_float(overall.get('mean_bleu'))}",
        f"- ROUGE-1 F1: {format_float(overall.get('mean_rouge1_f1'))}",
        f"- ROUGE-2 F1: {format_float(overall.get('mean_rouge2_f1'))}",
        f"- ROUGE-L F1: {format_float(overall.get('mean_rougeL_f1'))}",
        f"- Official METEOR: {format_float(overall.get('mean_meteor'))}",
        f"- Raw BERTScore F1: {format_float(overall.get('mean_bertscore_raw_f1'))}",
        f"- Strict BERTScore F1: {format_float(overall.get('mean_bertscore_strict_f1'))}",
        f"- Exact Match: {format_float(overall.get('mean_exact_match_norm'))}",
        "",
        "Answered-only diagnostic evaluation:",
        f"- Answered questions: {int(answered_summary.get('num_questions', 0)) if answered_summary else 0}",
        f"- BLEU: {format_float(answered_summary.get('mean_bleu') if answered_summary else None)}",
        f"- ROUGE-L F1: {format_float(answered_summary.get('mean_rougeL_f1') if answered_summary else None)}",
        f"- Official METEOR: {format_float(answered_summary.get('mean_meteor') if answered_summary else None)}",
        f"- Raw BERTScore F1: {format_float(answered_summary.get('mean_bertscore_raw_f1') if answered_summary else None)}",
        f"- Strict BERTScore F1: {format_float(answered_summary.get('mean_bertscore_strict_f1') if answered_summary else None)}",
        "",
        "Scenario-level all-question metrics:",
    ]

    for _, row in all_scenario.iterrows():
        lines.append(
            f"- {row['scenario']}: questions={int(row['num_questions'])}, "
            f"refusals={int(row['chatgpt_refusal_count'])}, "
            f"BLEU={format_float(row.get('mean_bleu'))}, "
            f"ROUGE-L={format_float(row.get('mean_rougeL_f1'))}, "
            f"METEOR={format_float(row.get('mean_meteor'))}, "
            f"Raw-BERT-F1={format_float(row.get('mean_bertscore_raw_f1'))}, "
            f"Strict-BERT-F1={format_float(row.get('mean_bertscore_strict_f1'))}"
        )

    lines += [
        "",
        "Qualitative analysis files:",
        "- chatgpt_refusals_for_qualitative_analysis.csv",
        "- low_score_questions_for_qualitative_analysis.csv",
        "- low_score_answered_only_questions_for_qualitative_analysis.csv",
        "- per_question_scores.csv",
    ]
    (out_dir / "paper_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ChatGPT PCAP-only answers against reference QA answers using final answer-quality metrics."
    )
    parser.add_argument("--reference-csv", default="canonical_data/reference_answers.csv")
    parser.add_argument("--prediction-csv", default="canonical_data/chatgpt_pcap_only_predictions.csv")
    parser.add_argument("--output-dir", default="results/chatgpt_pcap_only_final_v7")
    parser.add_argument("--model-name", default="ChatGPT_PCAP_only")
    parser.add_argument("--use-bertscore", action="store_true")
    parser.add_argument(
        "--raw-bertscore-model",
        default=None,
        help="Optional model_type for raw BERTScore. Default None matches the provided reference code, which uses lang='en' only.",
    )
    parser.add_argument("--strict-bertscore-model", default="roberta-large")
    parser.add_argument("--raw-bertscore-batch-size", type=int, default=16)
    parser.add_argument("--strict-bertscore-batch-size", type=int, default=8)
    parser.add_argument(
        "--download-nltk",
        action="store_true",
        default=True,
        help="Attempt to download WordNet/OMW if official METEOR resources are missing. Enabled by default.",
    )
    parser.add_argument(
        "--no-download-nltk",
        action="store_false",
        dest="download_nltk",
        help="Do not attempt NLTK downloads; fail if official METEOR resources are missing.",
    )
    parser.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help="Do not fail if prediction keys or question text do not exactly match the reference. Use only for debugging.",
    )
    args = parser.parse_args()

    check_metric_dependencies(use_bertscore=args.use_bertscore)
    meteor_status = ensure_official_meteor(download_nltk=args.download_nltk)

    ref_df = load_csv(Path(args.reference_csv))
    pred_df = load_csv(Path(args.prediction_csv))
    validation_report = validate_inputs(ref_df, pred_df, strict=not args.allow_input_mismatch)

    scored, bert_status = build_scored_dataframe(
        ref_df=ref_df,
        pred_df=pred_df,
        model_name=args.model_name,
        use_bertscore=args.use_bertscore,
        raw_bertscore_batch_size=args.raw_bertscore_batch_size,
        strict_bertscore_batch_size=args.strict_bertscore_batch_size,
        raw_bertscore_model=args.raw_bertscore_model,
        strict_bertscore_model=args.strict_bertscore_model,
    )

    write_outputs(
        scored=scored,
        out_dir=Path(args.output_dir),
        validation_report=validation_report,
        meteor_status=meteor_status,
        bert_status=bert_status,
        use_bertscore=args.use_bertscore,
    )

    print(f"Done. Results written to: {args.output_dir}")
    print(f"Questions evaluated: {len(scored)}")
    print(f"ChatGPT refusals: {int(scored['chatgpt_refused'].sum())}")
    print(f"Answered-only questions: {int(scored['answered_non_refusal'].sum())}")
    print("Main outputs:")
    print(f"  {args.output_dir}/overall_metrics_all_questions.csv")
    print(f"  {args.output_dir}/scenario_metrics_all_questions.csv")
    print(f"  {args.output_dir}/category_metrics_all_questions.csv")
    print(f"  {args.output_dir}/overall_metrics_answered_only.csv")
    print(f"  {args.output_dir}/scenario_metrics_answered_only.csv")
    print(f"  {args.output_dir}/category_metrics_answered_only.csv")
    print(f"  {args.output_dir}/per_question_scores.csv")
    print(f"  {args.output_dir}/metric_status.json")
    print(f"  {args.output_dir}/paper_summary.txt")


if __name__ == "__main__":
    main()



"""
nohup python scripts/evaluate_frontier_pcap_baseline.py \
  --reference-csv canonical_data/reference_answers.csv \
  --prediction-csv canonical_data/chatgpt_pcap_only_predictions.csv \
  --output-dir results/chatgpt_pcap_only_full_metrics \
  --model-name ChatGPT_PCAP_only \
  --use-bertscore \
  &


"""