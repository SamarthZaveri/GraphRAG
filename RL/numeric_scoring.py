"""Deterministic scalar-answer scoring; not FinQA program execution accuracy."""
from __future__ import annotations
import math
import re

NUMBER = r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)"
UNITS = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
# v2: descriptive unit words may follow the number ("71 thousand barrels per day"); a bare number
# may be a percent when the published answer is one; a scale word matching the filing's own units is accepted.
VERSION = "finqa_scalar_v2"
PERCENT_UNITS = ("%", "percent", "percentage points")


def parse_scalar(text):
    text = str(text).strip().lower().replace(chr(0x2212), "-")
    if len(re.findall(r"FINAL_NUMERIC_ANSWER:", text, re.I)) > 1:
        return None
    match = re.search(r"FINAL_NUMERIC_ANSWER:\s*([^\n]+)", text, re.I)
    if match:
        text = match.group(1).strip()
    # No extraction of arbitrary numbers from prose, equations or citations.
    boolean = text.strip(". ")
    if boolean in ("yes", "no"):
        return boolean, "boolean", None
    match = re.fullmatch(r"\s*[$\u20ac\u00a3]?\s*(\()?\s*(" + NUMBER + r")\s*(\))?\s*(%|percent|percentage points|ratio|thousand|million|billion|dollars|usd)?(?:\s+[a-z][a-z\s/-]*)?\s*\.?", text)
    if not match or bool(match[1]) != bool(match[3]):
        return None
    value = float(match[2].replace(",", ""))
    if match[1]:
        value = -abs(value)
    if not math.isfinite(value):
        return None
    decimals = len(match[2].split(".")[1]) if "." in match[2] else 0
    return value, match[4] or "number", decimals


def score_numeric(answer, question):
    parsed = parse_scalar(answer)
    gold = parse_scalar(question["reference_answer"])
    if not parsed or not gold:
        return {"score": 0., "rationale": "No unambiguous scalar final answer or unsupported gold format", "metric": VERSION}
    predicted, unit, _ = parsed
    expected, _, _ = gold
    if isinstance(expected, str):
        correct = predicted == expected
        return {"score": 5. if correct else 0., "rationale": "Boolean gold comparison", "metric": VERSION}
    if isinstance(predicted, str):
        return {"score": 0., "rationale": "Expected a number", "metric": VERSION}
    human = parse_scalar(question.get("human_answer", ""))
    human_unit = human[1] if human else "number"
    program = question.get("reference_program", "") or ""
    # FinQA exe_ans is commonly a ratio even when the human answer is percent.
    final_operation = program.rsplit(")", 2)[-2] if ")" in program else program
    percentage_points_gold = bool(re.search(r"multiply\([^)]*(?:const_100|\b100)\b", final_operation))
    gold_ratio = human_unit in PERCENT_UNITS and not percentage_points_gold
    candidates = [predicted]
    if unit in PERCENT_UNITS:
        if not gold_ratio and not percentage_points_gold:
            return {"score": 0., "rationale": "Unexpected percent unit for gold value", "metric": VERSION}
        if gold_ratio:
            candidates = [predicted / 100]
    elif unit in UNITS:
        reference_scale = UNITS.get(human_unit)
        if reference_scale is None:
            # Only infer a scale explicitly requested in the question.
            wording = question["question"].lower()
            reference_scale = next((factor for name, factor in UNITS.items() if re.search(r"in (?:\$\s*)?" + name + r"s?\b", wording)), None)
        # With no scale stated anywhere, the gold is in the filing's own units, which the model may name.
        candidates = [predicted * UNITS[unit] / reference_scale] if reference_scale else [predicted]
    elif human_unit in UNITS and unit in ("dollars", "usd"):
        candidates = [predicted / UNITS[human_unit]]
    elif gold_ratio:
        # A bare number for a percent answer: accept it read either as the ratio or as a percent.
        candidates = [predicted, predicted / 100]
    tolerance = max(1e-5, abs(expected) * .001)
    # Allow the published human answer's precision, without adopting its value.
    if human and isinstance(human[0], float):
        precision = .5 * 10 ** (-human[2])
        if gold_ratio:
            precision /= 100
        if human_unit == "number" or human_unit in PERCENT_UNITS:
            tolerance = max(tolerance, precision)
    predicted = min(candidates, key=lambda value: abs(value - expected))
    correct = abs(predicted - expected) <= tolerance
    return {"score": 5. if correct else 0., "rationale": "Numeric match" if correct else "Numeric mismatch",
            "prediction": predicted, "gold": expected, "tolerance": tolerance, "metric": VERSION}
