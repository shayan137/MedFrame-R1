import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description="Build an order file from the samples that were trainable in a run")
    parser.add_argument("--sample_log", type=str, required=True, help="Path to sample_log.jsonl of a run with online filtering")
    parser.add_argument("--output", type=str, required=True, help="Path of the order file, used as data.order_file")
    parser.add_argument("--seed", type=int, default=1, help="Seed for shuffling the rows")
    args = parser.parse_args()

    # A sample is kept if its group had mixed outcomes at least once
    row_ids = set()
    with open(args.sample_log, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if sample["outcome"] == "trainable":
                row_ids.add(sample["row_id"])

    row_ids = sorted(row_ids)
    random.Random(args.seed).shuffle(row_ids)
    with open(args.output, "w", encoding="utf-8") as f:
        for row_id in row_ids:
            f.write(json.dumps({"row_id": row_id}) + "\n")

    print(f"Saved {len(row_ids)} trainable samples to {args.output}")


if __name__ == "__main__":
    main()
