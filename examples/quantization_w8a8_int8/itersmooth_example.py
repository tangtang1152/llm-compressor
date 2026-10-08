"""IterSmooth followed by W8A8 calibration in separate independent epochs.

Example:
    python itersmooth_example.py --model /path/to/model --dataset calib.jsonl \
        --output-dir /path/to/output

The JSON/JSONL dataset must contain a ``text`` column.
"""

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.itersmooth import IterSmoothModifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--num-calibration-samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--gptq", action="store_true")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    quantizer = GPTQModifier if args.gptq else QuantizationModifier
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=[
            IterSmoothModifier(alpha=args.alpha),
            quantizer(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
        ],
        pipeline="independent",
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
