"""Read-only server calibration preflight. Never loads model weights."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from .diagnose import atomic_json


def inspect_data(
    dataset, tokenizer, samples=42, max_length=2688, *, return_tokens=False
):
    text = dataset.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
        if not rows or not all(isinstance(r, str) for r in rows):
            raise ValueError("historical JSON array must be nonempty list[str]")
        rows = [{"text": r} for r in rows]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if samples < 1 or len(rows) < samples:
        raise ValueError("not enough samples in the original calibration source")
    tokenized = []
    lengths = []
    for sample_id, row in enumerate(rows[:samples]):
        prompt = row.get("text")
        if prompt is None and "messages" in row:
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"empty/invalid sample {sample_id}")
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_token_type_ids=False,
        )
        if not encoded["input_ids"]:
            raise ValueError("empty tokenized sample")
        lengths.append(sum(encoded["attention_mask"]))
        tokenized.append(dict(sample_id=sample_id, **encoded))
    result = dict(
        dataset=str(dataset),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        samples_available=len(rows),
        samples_selected=samples,
        global_sample_ids=list(range(samples)),
        valid_tokens=sum(lengths),
        length_min=min(lengths),
        length_max=max(lengths),
        tokenized_sha256=hashlib.sha256(
            json.dumps(tokenized, sort_keys=True).encode()
        ).hexdigest(),
        preprocessing=dict(
            max_length=max_length,
            batch_size=1,
            truncation=True,
            add_special_tokens=False,
            return_token_type_ids=False,
            chat_template="messages only; add_generation_prompt=False",
            shuffle=False,
        ),
    )
    return (result, tokenized) if return_tokens else result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--dataset", type=Path, default=Path("/mnt/share/datasets/qwen3_cot_w4a4.json")
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=42)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
        result.update(inspect_data(a.dataset, tokenizer, a.samples))
        if a.samples == 42 and (
            result["valid_tokens"],
            result["length_min"],
            result["length_max"],
        ) != (91834, 1244, 2672):
            raise ValueError(
                "token lengths differ from historical 42-sample run; "
                "verify data/tokenizer before capture"
            )
        result["tokenizer_files_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in a.checkpoint.iterdir()
            if p.is_file()
            and (p.name.startswith("tokenizer") or p.name == "chat_template.jinja")
        }
        result["status"] = "data_preflight_passed_capture_not_validated"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        atomic_json(a.output / "data_preflight.json", result)


if __name__ == "__main__":
    main()
