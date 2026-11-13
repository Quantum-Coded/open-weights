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
from pathlib import Path

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


def batch_invariant() -> bool:
    """Kernel mode used for this process (pre-registered: on for every run, off only for the
    noise-floor reruns, which set VLLM_BATCH_INVARIANT=0)."""
    import os
    return bool(int(os.environ.get("VLLM_BATCH_INVARIANT", "1")))


def load_llm(model_dir: str, max_len: int, gpu_util: float | None = None):
    # Default 0.80: on the laptop the Windows desktop holds ~1 GB of the 6 GB, so only
    # ~4.95 GiB is free at startup and 0.90 (5.4 GiB) fails. Override with WSP_GPU_UTIL.
    import os
    # Batch-invariant kernels by default: plain greedy decoding changes with batch composition
    # (bf16 near-ties), so without this the same command can give different text. Must be set
    # before vLLM is imported.
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if batch_invariant() else "0"
    from vllm import LLM
    if gpu_util is None:
        gpu_util = float(os.environ.get("WSP_GPU_UTIL", "0.80"))
    # Attention backend pinned (auto-selection picks FLASH_ATTN on this GPU anyway); vLLM's
    # batch-invariant mode (env VLLM_BATCH_INVARIANT=1) refuses to start unless it is explicit.
    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=max_len,
              gpu_memory_utilization=gpu_util, enable_prefix_caching=True,
              attention_backend="FLASH_ATTN", seed=0)
    llm._wsp_max_len = max_len
    return llm


def free_llm(llm) -> None:
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()


KV_GROUP_FLAG = Path(__file__).resolve().parent / "kv_group.on"


def kv_capacity(llm) -> int | None:
    """KV-cache size in tokens, as the engine reported it at start-up (None if unknown)."""
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        return int(cc.num_gpu_blocks) * int(cc.block_size) if cc.num_gpu_blocks else None
    except Exception:
        return None


def _groups(llm, active: list[Gen], params, *, force: bool | None = None) -> list[tuple[int, int]]:
    """Split one chunk round into consecutive groups that fit the KV cache together.

    Added 13 Nov (Amendment 4, operational). On the 6 GB GPU the KV cache holds ~27k tokens, so in
    the last rounds of a 16k-token run at most one answer fits, and one llm.generate call over all of
    them made vLLM preempt and recompute 14k-token prefixes over and over (26 answers took 6 h 21 min).
    Every request carries its own seed and the kernels are batch-invariant, so the text of an answer
    does not depend on which other answers share its batch; kv_group_test.py checks this token for
    token before the flag file is created. Only used when the flag file exists (or force=True), when
    the round does not fit in the cache, and when at most 2 answers would fit at once.
    """
    n = len(active)
    on = KV_GROUP_FLAG.exists() if force is None else force
    cap = kv_capacity(llm) if on else None
    if not cap or n <= 1:
        return [(0, n)]
    need = [len(g.prompt_ids) + len(g.out_ids) + p.max_tokens for g, p in zip(active, params)]
    k = int(0.95 * cap) // max(need)
    if sum(need) <= cap or k > 2:
        return [(0, n)]
    k = max(k, 1)
    return [(i, min(i + k, n)) for i in range(0, n, k)]


def generate(llm, rows: list[dict], *, max_tokens: int, temperature: float,
             top_p: float, n_samples: int = 1, early_stop: bool = True,
             seed: int = 0, presence_penalty: float = 0.0) -> list[dict]:
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    eos = tok.eos_token_id
    max_ctx = getattr(llm, "_wsp_max_len", None) or 10**9
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
        outs = []
        for lo, hi in _groups(llm, active, params):
            outs.extend(llm.generate(prompts[lo:hi], params[lo:hi], use_tqdm=True))
        for g, o in zip(active, outs):
            c = o.outputs[0]
            g.out_ids.extend(c.token_ids)
            if g.first_flag is None and is_looping(g.out_ids):
                g.first_flag = len(g.out_ids)
            if c.finish_reason == "stop" or (g.out_ids and g.out_ids[-1] == eos):
                g.status = "finished"
            elif len(g.out_ids) >= max_tokens or len(g.prompt_ids) + len(g.out_ids) >= max_ctx:
                # max_ctx: a prompt longer than the 1,024-token allowance can reach the model's
                # context limit before max_tokens; vLLM then stops with "length" and one more
                # request would crash (8 Nov fix, Amendment 3). Counted as a cap.
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
