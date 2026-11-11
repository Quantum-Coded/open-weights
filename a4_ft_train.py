"""Amendment 4, fine-tune baseline step 2: LoRA supervised fine-tune on a4_data/ft_train.jsonl, merged to work/ft.

Hyperparameters are fixed before any run (common defaults, no tuning, no validation-based early stopping):
  LoRA r 16, alpha 32, dropout 0.05, on q/k/v/o/gate/up/down of every layer; adapters fp32, base bf16.
  AdamW lr 1e-4, weight decay 0, warmup 10 steps then cosine to 0, grad clip 1.0.
  2 epochs, 16 sequences per optimizer step (gradient accumulation), seed 0, gradient checkpointing.
  Loss: next-token cross-entropy on the response tokens only (prompt masked), EOS appended to each response.
The prompt is the same chat prompt the evaluation uses (common.build_prompt_ids).

  python a4_ft_train.py            # trains, logs loss, merges, saves work/ft (+ a4_data/ft_lora/ adapter)
"""
import json, math, random, shutil, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common import build_prompt_ids, read_jsonl

ROOT = Path(__file__).resolve().parent
BASE, OUT = ROOT / "work/base", ROOT / "work/ft"
R, ALPHA, DROP, LR, EPOCHS, ACCUM, WARMUP, CLIP, SEED = 16, 32, 0.05, 1e-4, 2, 16, 10, 1.0, 0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CE_CHUNK = 256


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE)
    data = []
    dry = "--dry" in sys.argv   # memory test only: 4 synthetic max-length sequences, 1 step, nothing saved
    src = read_jsonl(ROOT / "editor/data/practice/math_train.jsonl")[:4] if dry else read_jsonl(ROOT / "a4_data/ft_train.jsonl")
    for r in src:
        p = build_prompt_ids(tok, r)
        y = tok(r["response"] if not dry else (r["question"] + " ") * 400, add_special_tokens=False)["input_ids"]
        y = (y[:3071] if dry else y) + [tok.eos_token_id]
        data.append((p, y))
    n_tok = sum(len(y) for _, y in data)
    print(f"{len(data)} examples, {n_tok:,} response tokens, longest {max(len(p) + len(y) for p, y in data)}")

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROP,
                                              target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    trunk = model.base_model.model.model          # Qwen2Model with LoRA layers inside
    head = model.base_model.model.lm_head         # frozen
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps = EPOCHS * math.ceil(len(data) / ACCUM)

    def lr_at(s):
        return LR * (s + 1) / WARMUP if s < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, steps - WARMUP)))

    def ce_sum(h, tgt):
        return F.cross_entropy(head(h).float(), tgt, reduction="sum")

    rng = random.Random(SEED)
    log, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(EPOCHS):
        order = list(range(len(data)))
        rng.shuffle(order)
        for g0 in range(0, len(order), ACCUM):
            group = [data[i] for i in order[g0:g0 + ACCUM]]
            denom = sum(len(y) for _, y in group)
            tot = 0.0
            for p, y in group:
                ids = torch.tensor([p + y], device="cuda")
                h = trunk(input_ids=ids).last_hidden_state[0]
                hs, tg = h[len(p) - 1: len(p) + len(y) - 1], ids[0, len(p):]
                loss = 0.0
                for c in range(0, len(y), CE_CHUNK):
                    loss = loss + checkpoint(ce_sum, hs[c:c + CE_CHUNK], tg[c:c + CE_CHUNK], use_reentrant=False)
                (loss / denom).backward()
                tot += loss.item()
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(params, CLIP).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": ep, "loss": tot / denom, "grad_norm": gn, "lr": lr_at(step - 1),
                   "elapsed_s": round(time.time() - t0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    if dry:
        print(f"DRY RUN OK: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, {time.time() - t0:.0f} s")
        return
    (ROOT / "a4_data/ft_train_log.json").write_text(json.dumps(log, indent=1))
    model.save_pretrained(ROOT / "a4_data/ft_lora")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    merged.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    for f in ("generation_config.json",):
        if (BASE / f).exists():
            shutil.copy(BASE / f, OUT / f)
    print(f"saved {OUT}; training took {time.time() - t0:.0f} s for {step} optimizer steps")


if __name__ == "__main__":
    main()
