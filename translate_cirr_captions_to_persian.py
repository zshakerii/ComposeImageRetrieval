#!/usr/bin/env python3
"""Translate CIRR captions (English -> Persian) with NLLB + rule-based validation.

Pipeline:
    English caption -> NLLB translation -> rule-based validation
    -> retry (higher beams) on failure -> caption_fa_literal
    -> rule-based normalization -> caption_fa_retrieval

Outputs per item:
    - caption_fa_literal   : validated, faithful NLLB translation
    - caption_fa_retrieval : normalized retrieval-friendly variant of the literal one

Validation rules enforced on the literal translation:
    - non-empty translation
    - numbers preserved (digits, Persian digits, number words)
    - colors preserved
    - no leftover untranslated Latin text
    - concise output (length-ratio guard against added information)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --------------------------------------------------------------------------
# Basic utilities
# --------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# --------------------------------------------------------------------------
# Rule-based validator (applied to the literal translation)
# --------------------------------------------------------------------------

EN_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}

FA_NUMBER_WORDS = {
    1: ["یک"], 2: ["دو"], 3: ["سه"], 4: ["چهار"], 5: ["پنج"],
    6: ["شش", "شیش"], 7: ["هفت"], 8: ["هشت"], 9: ["نه"], 10: ["ده"],
    11: ["یازده"], 12: ["دوازده"], 13: ["سیزده"], 14: ["چهارده"],
    15: ["پانزده"], 16: ["شانزده"], 17: ["هفده"], 18: ["هجده"],
    19: ["نوزده"], 20: ["بیست"],
}

EN_FA_COLORS = {
    "red": ["قرمز", "سرخ"],
    "blue": ["آبی"],
    "green": ["سبز"],
    "yellow": ["زرد"],
    "black": ["سیاه", "مشکی"],
    "white": ["سفید"],
    "brown": ["قهوه‌ای", "قهوه ای"],
    "gray": ["خاکستری"],
    "grey": ["خاکستری"],
    "orange": ["نارنجی"],
    "pink": ["صورتی"],
    "purple": ["بنفش", "ارغوانی"],
    "gold": ["طلایی"],
    "golden": ["طلایی"],
    "silver": ["نقره‌ای", "نقره ای"],
    "beige": ["بژ", "کرم"],
}

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _extract_numbers_en(text: str) -> List[int]:
    lowered = text.lower()
    numbers = [int(m) for m in re.findall(r"\d+", lowered)]
    for word, value in EN_NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            numbers.append(value)
    return numbers


def _number_present_fa(value: int, fa_text: str) -> bool:
    western = str(value)
    if western in fa_text or western.translate(FA_DIGITS) in fa_text:
        return True
    return any(w in fa_text for w in FA_NUMBER_WORDS.get(value, []))


@dataclass
class ValidationResult:
    valid: bool
    issues: List[str] = field(default_factory=list)


def validate_translation(en_caption: str, fa_caption: str) -> ValidationResult:
    issues: List[str] = []
    fa = normalize_text(fa_caption)
    en = normalize_text(en_caption)

    if not fa:
        return ValidationResult(False, ["empty_translation"])

    for num in set(_extract_numbers_en(en)):
        if not _number_present_fa(num, fa):
            issues.append(f"missing_number:{num}")

    en_lower = en.lower()
    for color_en, fa_variants in EN_FA_COLORS.items():
        if re.search(rf"\b{color_en}\b", en_lower):
            if not any(v in fa for v in fa_variants):
                issues.append(f"missing_color:{color_en}")

    # Leftover untranslated Latin words (3+ letters) suggest a failed translation.
    if re.search(r"[A-Za-z]{3,}", fa):
        issues.append("latin_leftover")

    # Conciseness / added-information guard (approximate):
    # a faithful Persian caption should not be much longer than the English one.
    en_tokens = max(1, len(en.split()))
    fa_tokens = len(fa.split())
    if fa_tokens > max(6, round(2.5 * en_tokens)):
        issues.append(f"too_long:{fa_tokens}vs{en_tokens}")

    return ValidationResult(valid=not issues, issues=issues)

# --------------------------------------------------------------------------
# Retrieval-friendly normalization (rule-based, derived from the literal text)
# --------------------------------------------------------------------------

AR_TO_FA_CHARS = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "ؤ": "و", "إ": "ا", "أ": "ا"})
FA_TO_WESTERN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
AR_TO_WESTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# Arabic diacritics (fatha, kasra, damma, tanwin, sukun, shadda, superscript alef) + tatweel
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
ZWNJ = "\u200c"


def make_retrieval_caption(literal_fa: str) -> str:
    """Build a retrieval-friendly caption from the validated literal translation.

    Steps: unify Arabic/Persian characters, strip diacritics and tatweel,
    unify all digits to Western digits, replace half-space (ZWNJ) with a space,
    drop punctuation, and collapse whitespace. Content words are untouched, so
    numbers, colors, objects, and spatial relations are preserved as-is.
    """
    text = normalize_text(literal_fa)
    text = text.translate(AR_TO_FA_CHARS)
    text = DIACRITICS_RE.sub("", text)
    text = text.translate(FA_TO_WESTERN_DIGITS).translate(AR_TO_WESTERN_DIGITS)
    text = text.replace(ZWNJ, " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------------------------------
# NLLB translator
# --------------------------------------------------------------------------

class NLLBTranslator:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        num_beams: int,
        max_length: int,
        torch_threads: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch_threads:
            torch.set_num_threads(torch_threads)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids("pes_Arab")

    @torch.no_grad()
    def translate_batch(
        self, captions: List[str], num_beams: Optional[int] = None
    ) -> List[str]:
        inputs = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.forced_bos_token_id,
            max_length=self.max_length,
            num_beams=num_beams or self.num_beams,
        )
        return [
            normalize_text(t)
            for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]

    def translate_validated(self, caption: str, retry_beams: int) -> Dict[str, Any]:
        """Translate one caption, retry with stronger beam search if validation fails.

        Returns {"literal": str, "retrieval": str, "valid": bool, "issues": [...]}.
        """
        first = self.translate_batch([caption])[0]
        result = validate_translation(caption, first)
        if result.valid:
            return make_cache_entry(first, valid=True, issues=[])

        retried = self.translate_batch([caption], num_beams=retry_beams)[0]
        retry_result = validate_translation(caption, retried)
        if retry_result.valid:
            return make_cache_entry(retried, valid=True, issues=[])

        # Keep the candidate with fewer issues; flag it for manual review.
        best_text, best_issues = (
            (retried, retry_result.issues)
            if len(retry_result.issues) < len(result.issues)
            else (first, result.issues)
        )
        return make_cache_entry(best_text, valid=False, issues=best_issues)

# --------------------------------------------------------------------------
# Cache (backward compatible with old str-valued and {"text": ...} caches)
# --------------------------------------------------------------------------

def make_cache_entry(literal: str, valid: bool, issues: List[str]) -> Dict[str, Any]:
    return {
        "literal": literal,
        "retrieval": make_retrieval_caption(literal),
        "valid": valid,
        "issues": issues,
    }


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    raw = load_json(path)
    cache: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, str):  # legacy v1: plain string
            cache[key] = make_cache_entry(value, valid=True, issues=[])
        elif "literal" in value:  # current format
            if not normalize_text(value.get("retrieval")):
                value["retrieval"] = make_retrieval_caption(value["literal"])
            cache[key] = value
        else:  # legacy v2: {"text", "valid", "issues"}
            cache[key] = make_cache_entry(
                value.get("text", ""),
                valid=value.get("valid", True),
                issues=value.get("issues", []),
            )
    return cache


def save_cache(cache: Dict[str, Dict[str, Any]], path: Path) -> None:
    atomic_save_json(cache, path)

# --------------------------------------------------------------------------
# Dataset handling
# --------------------------------------------------------------------------

def collect_unique_missing_captions(
    data: List[dict],
    cache: Dict[str, Dict[str, Any]],
    literal_field: str,
    retrieval_field: str,
) -> List[str]:
    seen, missing = set(), []
    for item in data:
        caption = normalize_text(item.get("caption"))
        if not caption:
            continue
        if normalize_text(item.get(literal_field)) and normalize_text(
            item.get(retrieval_field)
        ):
            continue
        key = sha256_text(caption)
        if key in cache or key in seen:
            continue
        seen.add(key)
        missing.append(caption)
    return missing


def add_translations(
    data: List[dict],
    cache: Dict[str, Dict[str, Any]],
    literal_field: str,
    retrieval_field: str,
) -> int:
    """Fill both output fields; never overwrite existing non-empty values."""
    added = 0
    for item in data:
        caption = normalize_text(item.get("caption"))
        entry = cache.get(sha256_text(caption))
        if not entry or not entry["literal"]:
            continue
        changed = False
        if not normalize_text(item.get(literal_field)):
            item[literal_field] = entry["literal"]
            changed = True
        if not normalize_text(item.get(retrieval_field)):
            item[retrieval_field] = entry["retrieval"]
            changed = True
        if changed:
            added += 1
    return added


def build_validation_report(
    data: List[dict],
    cache: Dict[str, Dict[str, Any]],
) -> List[dict]:
    report, reported = [], set()
    for item in data:
        caption = normalize_text(item.get("caption"))
        key = sha256_text(caption)
        entry = cache.get(key)
        if entry and not entry.get("valid", True) and key not in reported:
            reported.add(key)
            report.append(
                {
                    "caption": caption,
                    "caption_fa_literal": entry["literal"],
                    "caption_fa_retrieval": entry["retrieval"],
                    "issues": entry["issues"],
                }
            )
    return report

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate CIRR captions to Persian (literal + retrieval) with NLLB + validation."
    )
    parser.add_argument("--input_json", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--cache_path", type=Path, default=None)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--literal_field", default="caption_fa_literal")
    parser.add_argument("--retrieval_field", default="caption_fa_retrieval")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--retry_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--torch_threads", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    args = parser.parse_args()

    cache_path = args.cache_path or args.output_json.with_name(
        args.output_json.stem + "_translations_cache.json"
    )
    report_path = args.output_json.with_name(
        args.output_json.stem + "_validation_report.json"
    )

    data = load_json(args.input_json)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dicts with a 'caption' field.")

    cache = load_cache(cache_path)
    missing = collect_unique_missing_captions(
        data, cache, args.literal_field, args.retrieval_field
    )
    print(f"Items: {len(data)} | unique captions to translate: {len(missing)}")

    if missing:
        translator = NLLBTranslator(
            model_name=args.model,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            max_length=args.max_length,
            torch_threads=args.torch_threads,
        )
        done = 0
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start : start + args.batch_size]
            translations = translator.translate_batch(batch)
            for caption, fa in zip(batch, translations):
                result = validate_translation(caption, fa)
                if result.valid:
                    cache[sha256_text(caption)] = make_cache_entry(
                        fa, valid=True, issues=[]
                    )
                else:
                    cache[sha256_text(caption)] = translator.translate_validated(
                        caption, retry_beams=args.retry_beams
                    )
                done += 1
                if done % args.save_every == 0:
                    save_cache(cache, cache_path)
                    print(f"  translated {done}/{len(missing)} (cache saved)")
        save_cache(cache, cache_path)

    added = add_translations(data, cache, args.literal_field, args.retrieval_field)
    atomic_save_json(data, args.output_json)

    report = build_validation_report(data, cache)
    atomic_save_json(report, report_path)

    print(f"Filled literal+retrieval for {added} items -> {args.output_json}")
    print(f"Flagged for manual review: {len(report)} -> {report_path}")


if __name__ == "__main__":
    main()
