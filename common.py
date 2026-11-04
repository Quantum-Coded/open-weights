"""Shared helpers: paths, prompts, answer extraction, grading, loop detection.

Used by eval.py (outside the editor folder) and editor/fp.py (inside it).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDITOR = ROOT / "editor"
PRACTICE_DIR = EDITOR / "data" / "practice"
SEALED_DIR = ROOT / "sealed"          # moved out of the project while Claude edits
RESULTS_DIR = ROOT / "results"
WORK_DIR = ROOT / "work"              # merged model copies (base + patch)

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# DeepSeek's usage notes for R1 distills: no system prompt, put instructions in
# the user turn, and for math ask for the final answer in \boxed{}.
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MCQ_SUFFIX = ("\nPlease reason step by step, and put the letter of the correct option "
              "within \\boxed{}.")


# ----------------------------------------------------------------------------- io
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def text_hash(s: str) -> str:
    norm = re.sub(r"\s+", " ", s.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------ prompts
def user_message(row: dict) -> str:
    if row["kind"] == "mcq":
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(row["options"]))
        return f"{row['question']}\n\n{opts}{MCQ_SUFFIX}"
    return row["question"] + MATH_SUFFIX


def build_prompt_ids(tokenizer, row: dict) -> list[int]:
    msgs = [{"role": "user", "content": user_message(row)}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # DeepSeek recommends forcing the response to open with "<think>\n".
    # Newer chat templates already add it; add it only if missing.
    if not text.rstrip().endswith("<think>"):
        text += "<think>\n"
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------- answer extraction
def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}), with brace matching."""
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None
    i = text.index("{", idx) + 1
    depth, start = 1, i
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None  # unbalanced (answer was cut off)


def answer_part(text: str) -> str:
    """Text after the reasoning block; whole text if </think> never appeared."""
    return text.split("</think>", 1)[1] if "</think>" in text else text


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
    try:
        return float(s)
    except ValueError:
        m = _NUM.findall(s)
        return float(m[-1].replace(",", "")) if m else None


# ------------------------------------------------------------------------ grading
def grade(row: dict, response: str) -> bool:
    """Automatic grading. No LLM judge anywhere."""
    if "</think>" not in response:      # never finished thinking -> no answer
        return False
    ans = answer_part(response)
    if row["kind"] == "mcq":
        b = last_boxed(ans)
        if b is None:
            return False
        b = re.sub(r"\\(?:text|textbf|mathrm)\{([^}]*)\}", r"\1", b)
        letters = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", b)
        return bool(letters) and letters[0] == row["answer"]
    if row["kind"] == "gsm8k":
        b = last_boxed(ans)
        pred = _to_float(b) if b is not None else (
            _to_float(_NUM.findall(ans)[-1]) if _NUM.findall(ans) else None)
        gold = _to_float(row["answer"])
        return pred is not None and gold is not None and abs(pred - gold) < 1e-6
    # kind == "math": symbolic check with the math-verify package
    from math_verify import parse, verify
    gold = parse("$" + row["answer"] + "$")
    pred = parse(ans)
    try:
        return bool(verify(gold, pred))
    except Exception:
        return False


# -------------------------------------------------------------- loop detection
LOOP_WINDOW = 2048     # look at the last N generated tokens
LOOP_NGRAM = 20        # an n-gram this long repeating verbatim is not normal prose
LOOP_FRAC = 0.60       # share of window positions whose n-gram already appeared


def repeated_fraction(tokens: list[int], window: int = LOOP_WINDOW, n: int = LOOP_NGRAM) -> float:
    w = tokens[-window:]
    if len(w) < n * 2:
        return 0.0
    seen: set[tuple] = set()
    repeats = 0
    total = len(w) - n + 1
    for i in range(total):
        g = tuple(w[i:i + n])
        if g in seen:
            repeats += 1
        else:
            seen.add(g)
    return repeats / total


def is_looping(tokens: list[int], window: int = LOOP_WINDOW, n: int = LOOP_NGRAM,
               frac: float = LOOP_FRAC) -> bool:
    """True once the tail of a generation is dominated by verbatim repetition.

    Thresholds are pre-registered in eval_plan.md and validated in the pilot
    (run with --no-early-stop and check how often a flagged answer would have
    finished correctly anyway).
    """
    if len(tokens) < window:
        return False
    return repeated_fraction(tokens, window, n) >= frac
