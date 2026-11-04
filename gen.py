"""Chunked generation with vLLM and early loop kill.

Each answer is generated CHUNK tokens at a time. After every chunk, finished
answers (EOS) leave the batch, answers whose tail is a verbatim loop are
stopped and marked "loop", and the rest continue from where they stopped
(vLLM's prefix cache makes the continuation cheap). With greedy decoding this
gives the same text as one long call; with sampling both arms use the same
procedure and seeds, so the comparison stays paired.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

from common import build_prompt_ids, is_looping, text_hash

CHUNK = 2048


@dataclass
class Gen:
    row: dict
    sample: int
    prompt_ids: list[int]
    out_ids: list[int] = field(default_factory=list)
    status: str = "running"      # running | finished | loop | cap
    first_flag: int | None = None  # token count when the detector first fired


def load_llm(model_dir: str, max_len: int, gpu_util: float = 0.90):
    from vllm import LLM
    return LLM(model=model_dir, dtype="bfloat16", max_model_len=max_len,
               gpu_memory_utilization=gpu_util, enable_prefix_caching=True,
               seed=0)


def free_llm(llm) -> None:
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def generate(llm, rows: list[dict], *, max_tokens: int, temperature: float,
             top_p: float, n_samples: int = 1, early_stop: bool = True,
             seed: int = 0, presence_penalty: float = 0.0) -> list[dict]:
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    eos = tok.eos_token_id
    gens = [Gen(row=r, sample=s, prompt_ids=build_prompt_ids(tok, r))
            for r in rows for s in range(n_samples)]

    while True:
        active = [g for g in gens if g.status == "running"]
        if not active:
            break
        prompts, params = [], []
        for g in active:
            prompts.append({"prompt_token_ids": g.prompt_ids + g.out_ids})
            step = min(CHUNK, max_tokens - len(g.out_ids))
            # Per-answer seed, shifted per chunk so continuations differ from
            # the first chunk's random stream (deterministic across arms).
            # (hashlib, not hash(): Python's hash() changes between processes)
            s = (seed * 1_000_003 + int(text_hash(f"{g.row['id']}|{g.sample}"), 16) % 1_000_003
                 + len(g.out_ids))
            params.append(SamplingParams(
                max_tokens=step, temperature=temperature,
                top_p=top_p if temperature > 0 else 1.0,
                presence_penalty=presence_penalty,
                seed=None if temperature == 0 else s,
                skip_special_tokens=False))
        outs = llm.generate(prompts, params, use_tqdm=True)
        for g, o in zip(active, outs):
            c = o.outputs[0]
            g.out_ids.extend(c.token_ids)
            if g.first_flag is None and is_looping(g.out_ids):
                g.first_flag = len(g.out_ids)
            if c.finish_reason == "stop" or (g.out_ids and g.out_ids[-1] == eos):
                g.status = "finished"
            elif len(g.out_ids) >= max_tokens:
                g.status = "cap"
            elif early_stop and g.first_flag is not None:
                g.status = "loop"

    results = []
    for g in gens:
        text = tok.decode(g.out_ids, skip_special_tokens=True)
        # A run that hit the cap is also checked for looping, for reporting.
        loop_flag = g.status == "loop" or (g.status == "cap" and is_looping(g.out_ids))
        results.append({"id": g.row["id"], "sample": g.sample, "status": g.status,
                        "loop": loop_flag, "first_flag": g.first_flag,
                        "n_tokens": len(g.out_ids), "response": text})
    return results
