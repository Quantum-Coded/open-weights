"""Download and split the data.

  python prepare_data.py

Writes
  editor/data/practice/math_train.jsonl, gsm8k_train.jsonl   (Claude may use these)
  sealed/math500.jsonl, gsm8k_test400.jsonl, mmlupro300.jsonl (Claude must never see these)

MATH train problems that duplicate a MATH-500 problem are dropped.
"""
import random

from datasets import load_dataset

from common import PRACTICE_DIR, SEALED_DIR, last_boxed, text_hash, write_jsonl

SEED = 1234
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def main():
    rng = random.Random(SEED)

    # ---- sealed test sets
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math500 = [{"id": f"m500-{i}", "kind": "math", "question": r["problem"], "answer": r["answer"],
                "subject": r["subject"], "level": r["level"]} for i, r in enumerate(m500)]
    test_hashes = {text_hash(r["question"]) for r in math500}

    gsm_test = list(load_dataset("openai/gsm8k", "main", split="test"))
    idx = sorted(rng.sample(range(len(gsm_test)), 400))
    gsm400 = [{"id": f"gsm-{i}", "kind": "gsm8k", "question": gsm_test[i]["question"],
               "answer": gsm_test[i]["answer"].split("####")[-1].strip()} for i in idx]
    test_hashes |= {text_hash(r["question"]) for r in gsm400}

    mp = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
    idx = sorted(rng.sample(range(len(mp)), 300))
    mmlu = [{"id": f"mmlup-{i}", "kind": "mcq", "question": mp[i]["question"],
             "options": mp[i]["options"], "answer": mp[i]["answer"],
             "category": mp[i]["category"]} for i in idx]

    write_jsonl(SEALED_DIR / "math500.jsonl", math500)
    write_jsonl(SEALED_DIR / "gsm8k_test400.jsonl", gsm400)
    write_jsonl(SEALED_DIR / "mmlupro300.jsonl", mmlu)

    # ---- practice sets (training splits only)
    math_train, dropped = [], 0
    for subj in MATH_SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", subj, split="train")):
            ans = last_boxed(r["solution"])
            if ans is None:
                continue
            if text_hash(r["problem"]) in test_hashes:
                dropped += 1
                continue
            math_train.append({"id": f"mtrain-{subj}-{i}", "kind": "math", "question": r["problem"],
                               "answer": ans, "subject": subj, "level": r["level"]})
    gsm_train = [{"id": f"gtrain-{i}", "kind": "gsm8k", "question": r["question"],
                  "answer": r["answer"].split("####")[-1].strip()}
                 for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train"))]

    write_jsonl(PRACTICE_DIR / "math_train.jsonl", math_train)
    write_jsonl(PRACTICE_DIR / "gsm8k_train.jsonl", gsm_train)

    print(f"sealed: math500={len(math500)} gsm8k={len(gsm400)} mmlu_pro={len(mmlu)}")
    print(f"practice: math_train={len(math_train)} (dropped {dropped} overlaps) gsm8k_train={len(gsm_train)}")


if __name__ == "__main__":
    main()
