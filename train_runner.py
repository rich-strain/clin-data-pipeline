"""Stage E — LoRA fine-tune of a small open model on Stage D's train/val JSONL.

Runs locally (Apple Silicon MPS), not in the deployed Streamlit app —
Streamlit Cloud has no GPU (CLAUDE.md Resolved decisions #3). This script,
its config, and its real results (loss curve, before/after samples) are
committed to `training_results/` so the deployed app can display an actual
completed run rather than re-running training live.

**Base model: Qwen2.5-0.5B-Instruct.** Considered against the alternatives
in its weight class:

- **TinyLlama-1.1B-Chat** — older base (mid-2023), noticeably weaker
  instruction-following in practice, and 2x the parameter count for worse
  quality; no reason to prefer it here.
- **SmolLM2-360M-Instruct** — smaller and faster, but 0.5B was not a
  meaningful download/runtime burden on this hardware (see below), and the
  larger model gives the fine-tune a better chance of actually picking up
  the JSON-extraction pattern rather than struggling with base capability.
- **Qwen2.5-0.5B-Instruct** (chosen) — modern (late-2024), already
  instruction-tuned (so the fine-tune only needs to shift behavior toward
  this specific extraction format, not teach instruction-following from
  scratch), single ~988 MB safetensors file (confirmed via the HF Hub API
  before downloading — a non-issue on this connection/timeline), and a
  standard Qwen2 attention architecture (`q_proj`/`k_proj`/`v_proj`/
  `o_proj`) that's a well-trodden LoRA target.

**No 4-bit/8-bit quantization.** `bitsandbytes` doesn't support MPS well —
skipped entirely rather than fighting a poorly-supported path. At 0.5B
parameters, fp16 LoRA fits comfortably in memory on this hardware without
needing quantization at all, so nothing is lost by skipping it.

**LoRA config:** `r=8, alpha=16` (the common `alpha = 2*r` heuristic),
`target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` (attention
projections only, not the MLP) — deliberately modest rather than maximal.
With only 139 training examples, a higher rank or MLP-inclusive target set
would add trainable capacity this dataset can't productively use; it would
more easily memorize/overfit the specific 139 examples rather than learn
the general extraction pattern. `dropout=0.05` for mild regularization at
this data scale.

**Prompt format:** the base model's own ChatML template
(`tokenizer.apply_chat_template`) — system message describing the
extraction task, user message = the instruction (redacted note text),
assistant message = the response (JSON string) being trained on. Loss is
masked to the assistant continuation only (labels for the system+user
prefix are set to `-100`) — the model should learn to *produce* the
extraction, not to predict the note text it's reading, which would just be
next-token-prediction noise unrelated to the actual task.

**Training loop is a plain PyTorch loop, not `transformers.Trainer`.**
`Trainer` would work, but a manual loop keeps every step (batching,
loss-masking, MPS device placement, per-epoch train/val loss) directly
inspectable in ~100 lines rather than mediated through `Trainer`'s
configuration surface — appropriate for a script whose job is partly to
*demonstrate* the mechanics, not just produce a checkpoint.

**Honesty about what this run demonstrates.** 139 training examples is a
small dataset by real ML standards, and a 0.5B model fine-tuned on it for a
handful of epochs is not a claim of production-quality extraction accuracy.
What this run *does* demonstrate, correctly and on real hardware: data
loading from the actual pipeline output, chat-template formatting with
correct loss masking, LoRA adapter attachment, a real training loop with
declining loss on real (if modest) data, adapter checkpointing, and
inference with vs. without the adapter to show a visible behavioral
difference. See `training_results/config.json` for the actual final loss
numbers from the run that produced the committed adapter — not
placeholders.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "splits" / "train.jsonl"
VAL_PATH = DATA_DIR / "splits" / "val.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "training_results"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a clinical data extraction assistant. Extract the patient's "
    "date of birth, diagnoses, medications, and vitals from the clinical "
    "note below. Respond with only a single JSON object with keys "
    "date_of_birth, diagnoses, medications, vitals — no other text."
)

LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

NUM_EPOCHS = 6
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_LENGTH = 768

# Colorblind-safe (Okabe-Ito): blue for train, amber for val.
COLOR_TRAIN = "#0072B2"
COLOR_VAL = "#E69F00"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def build_messages(instruction, response=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def encode_example(tokenizer, instruction, response, max_length):
    """Tokenize one (instruction, response) pair with labels masked to the
    assistant continuation only."""
    prompt_text = tokenizer.apply_chat_template(
        build_messages(instruction), tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        build_messages(instruction, response), tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_length]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_token_id):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def batches(examples, batch_size, shuffle, generator=None):
    order = torch.randperm(len(examples), generator=generator).tolist() if shuffle else list(range(len(examples)))
    for i in range(0, len(order), batch_size):
        yield [examples[j] for j in order[i:i + batch_size]]


def run_eval(model, examples, pad_token_id, device, batch_size):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in batches(examples, batch_size, shuffle=False):
            enc = collate(batch, pad_token_id)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, use_cache=False)
            n_target_tokens = (enc["labels"] != -100).sum().item()
            total_loss += out.loss.item() * n_target_tokens
            total_tokens += n_target_tokens
            del out
            if device.type == "mps":
                torch.mps.empty_cache()
    model.train()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, instruction, device, max_new_tokens=320):
    prompt_text = tokenizer.apply_chat_template(
        build_messages(instruction), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune a small model on Stage D's train/val JSONL.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--n-samples", type=int, default=3, help="Number of before/after val samples to generate")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device} (MPS available: {torch.backends.mps.is_available()})")

    train_data = read_jsonl(TRAIN_PATH)
    val_data = read_jsonl(VAL_PATH)
    print(f"Loaded {len(train_data)} train / {len(val_data)} val examples")

    print(f"Loading base model {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
    base_model.to(device)

    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # Gradient checkpointing trades recompute for memory -- worth it here:
    # Qwen2.5's 151,936-token vocab makes the final logits tensor large
    # relative to this model's hidden size, and MPS's allocator handles
    # that less gracefully than CUDA's. Dataset is tiny, so the extra
    # recompute cost is negligible.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_encoded = [encode_example(tokenizer, ex["instruction"], ex["response"], MAX_LENGTH) for ex in train_data]
    val_encoded = [encode_example(tokenizer, ex["instruction"], ex["response"], MAX_LENGTH) for ex in val_data]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    gen = torch.Generator().manual_seed(42)

    print(f"\nTraining for {args.epochs} epochs (batch size {args.batch_size}, lr {args.lr}) on {device} ...")
    start = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_tokens = 0.0, 0
        for batch in batches(train_encoded, args.batch_size, shuffle=True, generator=gen):
            enc = collate(batch, tokenizer.pad_token_id)
            enc = {k: v.to(device) for k, v in enc.items()}

            optimizer.zero_grad()
            out = model(**enc, use_cache=False)
            n_target_tokens = (enc["labels"] != -100).sum().item()
            loss_value = out.loss.item()
            out.loss.backward()
            optimizer.step()

            del out, enc
            if device.type == "mps":
                torch.mps.empty_cache()

            epoch_loss += loss_value * n_target_tokens
            epoch_tokens += n_target_tokens

        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss = run_eval(model, val_encoded, tokenizer.pad_token_id, device, args.batch_size)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.1f}s")

    # --- Before/after generation samples (base vs. fine-tuned) -------------

    # Critical: must switch out of .train() mode before generating. In
    # .train() mode, gradient-checkpointing recomputation hooks and LoRA
    # dropout are both active; neither belongs in an inference call, and
    # combined with MPS the result was reproducibly garbage output (not a
    # subtle quality regression — complete token repetition) even from the
    # unmodified base model under `disable_adapter()`. Caught and confirmed
    # by isolating the exact difference against a working reproduction.
    model.eval()

    print(f"\nGenerating {args.n_samples} before/after samples ...")
    sample_records = val_data[:args.n_samples]
    samples = []
    for ex in sample_records:
        with model.disable_adapter():
            base_output = generate(model, tokenizer, ex["instruction"], device)
        tuned_output = generate(model, tokenizer, ex["instruction"], device)
        samples.append({
            "instruction": ex["instruction"],
            "ground_truth": ex["response"],
            "base_model_output": base_output,
            "fine_tuned_output": tuned_output,
        })

    # --- Save artifacts ------------------------------------------------

    args.out.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.out / "adapter"
    model.save_pretrained(adapter_dir)

    config_out = {
        "base_model": BASE_MODEL,
        "device": str(device),
        "lora_config": LORA_CONFIG,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": MAX_LENGTH,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "training_seconds": round(elapsed, 1),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "loss_history": history,
    }
    (args.out / "config.json").write_text(json.dumps(config_out, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", color=COLOR_TRAIN, label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", color=COLOR_VAL, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(
        f"LoRA fine-tune of {BASE_MODEL}\non {len(train_data)} train / {len(val_data)} val examples", fontsize=11
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "loss_curve.png", dpi=150)
    plt.close(fig)

    samples_md = ["# Before/after generation samples\n"]
    for i, s in enumerate(samples, 1):
        samples_md.append(f"## Sample {i}\n")
        samples_md.append(f"**Instruction:**\n```\n{s['instruction']}\n```\n")
        samples_md.append(f"**Ground truth response:**\n```json\n{s['ground_truth']}\n```\n")
        samples_md.append(f"**Base model (no adapter) output:**\n```\n{s['base_model_output']}\n```\n")
        samples_md.append(f"**Fine-tuned model (with adapter) output:**\n```\n{s['fine_tuned_output']}\n```\n")
    (args.out / "samples.md").write_text("\n".join(samples_md))
    (args.out / "samples.json").write_text(json.dumps(samples, indent=2))

    print(f"\nSaved adapter, config.json, loss_curve.png, samples.md/.json to {args.out}")


if __name__ == "__main__":
    main()
