import json
import os
import statistics


def load_evaluations(file_path):
    """Load an eval JSONL file into a dict keyed by sample ID."""
    results = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results[result["id"]] = result
    return results


def merge_judges(judge_files, output_file):
    """Merge the eval files of several judges into one eval file.
    MIMIC tasks use the median of the 0-5 scores; all others use majority-vote binary scoring.
    """
    passes = {judge: load_evaluations(file_path) for judge, file_path in judge_files.items()}
    for judge, results in passes.items():
        if not results:
            raise FileNotFoundError(f"No evaluation results found for judge '{judge}' in {judge_files[judge]}")

    merged = []
    num_judged, num_split = 0, 0
    for sample_id in sorted(set().union(*passes.values())):
        results = {judge: passes[judge][sample_id] for judge in passes if sample_id in passes[judge]}
        merged_result = dict(next(iter(results.values())))

        # Rule-based scores do not depend on the judge and must agree across the files
        if merged_result.get("scored_by") != "judge":
            rule_scores = {result["evaluation"]["score"] for result in results.values()}
            if len(rule_scores) > 1:
                raise ValueError(f"Rule-based scores of {sample_id} differ across judges: {rule_scores}")
            merged.append(merged_result)
            continue

        # A judge that gave no score for the sample is left out of the vote
        votes = {
            judge: result["evaluation"]["score"]
            for judge, result in results.items()
            if result.get("scored_by") == "judge" and result["evaluation"]["score"] is not None
        }
        if not votes:
            merged.append(merged_result)
            continue

        scores = list(votes.values())
        if 'mimic' in sample_id:
            score = statistics.median(scores)
        else:
            score = 1 if scores.count(1) * 2 > len(scores) else 0

        merged_result["evaluation"] = {"score": score}
        merged_result["judge_votes"] = votes
        merged_result["n_votes"] = len(scores)
        merged.append(merged_result)
        num_judged += 1
        num_split += len(set(scores)) > 1

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in merged:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Merged {len(merged)} samples from {len(passes)} judges into {output_file}")
    print(f"Scored by judges: {num_judged}, with disagreement between judges: {num_split}")
