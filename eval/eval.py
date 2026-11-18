import argparse
import sys
from tqdm import tqdm
import yaml
import os
import concurrent.futures
from functools import partial
import pandas as pd
from utils import *
from merge_judges import merge_judges

# Default values of the eval section in the config file
DEFAULT_CONFIG = {
    "model": None,
    "name": None,
    "tasks": ["all"],
    "tasks_file": "tasks.yaml",
    "output_dir": "results",
    "num_workers": 128,
    "system_prompt": None,
    "max_tokens": 2048,
    "request_timeout": 180,
    "tensor_parallel_size": 1,
    "data_parallel_size": 1,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.95,
    "enforce_eager": True,
    "seed": 0,
    "judge_server": "local",
    "judges": [{"name": "qwen3", "model": "Qwen/Qwen3-14B"}],
    "judge_tensor_parallel_size": 1,
    "judge_data_parallel_size": 1,
    "judge_rounds": 1,
    "judge_max_tokens": 4096,
    "judge_max_tokens_mimic": 16384,
    "judge_timeout": 600,
    "exact_max_words": 25,
    "binary_scope_words": 30,
    "binary_final_words": 12,
}


def str2bool(value):
    """Parse a boolean command line value."""
    if str(value).strip().lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if str(value).strip().lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on medical datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration YAML file (the eval section is used)')
    parser.add_argument('--generate', type=str2bool, default=False, help='Generate responses for the given tasks')
    parser.add_argument('--evaluate', type=str2bool, default=False, help='Evaluate the generated responses')
    parser.add_argument('--score', type=str2bool, default=False, help='Calculate scores for the evaluation results')
    args = parser.parse_args()

    # Load evaluation settings from the eval section of the config file
    with open(args.config, 'r') as file:
        file_config = yaml.safe_load(file).get("eval") or {}
    unknown_keys = set(file_config) - set(DEFAULT_CONFIG)
    if unknown_keys:
        print(f"Error: unknown keys in the eval section of {args.config}: {', '.join(sorted(unknown_keys))}")
        sys.exit(1)
    config = {**DEFAULT_CONFIG, **file_config}

    # Validate OpenRouter API key upfront
    if config["judge_server"] == 'openrouter' and not os.getenv('OPENROUTER_API_KEY'):
        print("Error: judge_server is set to 'openrouter' but OPENROUTER_API_KEY is not set.")
        print("Set it before running:")
        print("  export OPENROUTER_API_KEY=your_openrouter_key")
        sys.exit(1)

    # Load dataset configurations from YAML file
    try:
        if os.path.exists(config["tasks_file"]):
            with open(config["tasks_file"], 'r') as file:
                dataset_configs = yaml.safe_load(file)
                print(f"Loaded dataset configurations from {config['tasks_file']}")
        else:
            print(f"Configuration file {config['tasks_file']} not found.")
            return
    except Exception as e:
        print(f"Error loading configuration from {config['tasks_file']}: {e}")
        return

    # Build flat task list and config lookup from the nested YAML structure
    datasets_to_load = []
    configs_to_load = {}
    all_tasks = []
    for cat in dataset_configs.keys():
        all_tasks += list(dataset_configs[cat].keys())
        for task in dataset_configs[cat]:
            configs_to_load[task] = dataset_configs[cat][task]

    for task in config["tasks"]:
        if task in all_tasks:
            datasets_to_load.append(task)
        elif task == "all":
            datasets_to_load += all_tasks
            break
        elif task in dataset_configs:
            datasets_to_load += list(dataset_configs[task].keys())
        else:
            print(f"Warning: Unknown dataset '{task}'")

    if not datasets_to_load:
        print("No valid datasets selected. Available options:")
        print(f"- Individual datasets: {', '.join(all_tasks)}")
        print(f"- Groups: all, {', '.join(dataset_configs.keys())}")
        return

    # Set up output paths
    model_name = config["model"]
    model_name_underscored = config["name"] or convert_to_underscored(model_name)
    output_dir = os.path.join(config["output_dir"], model_name_underscored)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name_underscored}.jsonl")
    eval_file = os.path.join(output_dir, f"{model_name_underscored}_eval.jsonl")
    judge_files = {
        judge["name"]: os.path.join(output_dir, f"{model_name_underscored}_eval_{judge['name']}.jsonl")
        for judge in config["judges"]
    }

    print(f"Model: {model_name}")
    print(f"Selected datasets: {datasets_to_load}")
    print(f"Results will be saved to: {output_file}")

    candidate_server = False
    eval_server = False

    # --- Phase 1: Generate responses ---
    if args.generate:
        try:
            # Load all datasets once to determine total expected count
            all_samples = []
            for dataset_name in datasets_to_load:
                print(f"Loading dataset: {dataset_name}")
                all_samples += load_dataset_with_params(configs_to_load[dataset_name], dataset_name)
            total_expected = len(all_samples)
            prev_completed = -1
            no_progress_passes = 0

            while True:
                completed_ids = load_completed_ids(output_file)
                remaining = [s for s in all_samples if s["id"] not in completed_ids]

                if not remaining:
                    print(f"All {total_expected} samples generated.")
                    break

                print(f"Generating responses for {len(remaining)}/{total_expected} samples using {config['num_workers']} workers...")

                if not candidate_server:
                    candidate_server = start_vllm_server(model_name, config["tensor_parallel_size"], config["data_parallel_size"], config)

                if config["num_workers"] <= 1:
                    for sample in tqdm(remaining):
                        process_sample(sample, output_file, model_name, config)
                else:
                    worker_func = partial(process_sample, output_file=output_file, model_name=model_name, config=config)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=config["num_workers"]) as executor:
                        futures = [executor.submit(worker_func, sample) for sample in remaining]
                        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                            pass

                # Re-check completeness after this pass
                completed_count = len(load_completed_ids(output_file))
                if completed_count >= total_expected:
                    print(f"All {total_expected} samples generated.")
                    break
                else:
                    print(f"Generated {completed_count}/{total_expected} samples. Retrying missing ones...")
                    # Restart the server if a pass made no progress, give up after three such passes
                    if completed_count <= prev_completed:
                        no_progress_passes += 1
                        if candidate_server:
                            kill_server_process(candidate_server)
                            candidate_server = None
                        if no_progress_passes >= 3:
                            print("Giving up on the remaining samples after repeated passes without progress.")
                            break
                    else:
                        no_progress_passes = 0
                prev_completed = completed_count

        finally:
            if candidate_server:
                kill_server_process(candidate_server)

    # --- Phase 2: Evaluate responses with fixed rules and each LLM judge ---
    if args.evaluate:
        all_responses = load_jsonl(output_file)
        total_expected = len(all_responses)

        for judge in config["judges"]:
            eval_model = judge["model"]
            judge_file = judge_files[judge["name"]]
            print(f"Evaluating results using model: {eval_model}")
            prev_completed = -1
            no_progress_passes = 0
            num_passes = 0

            try:
                while True:
                    evaluated_ids = load_completed_ids(judge_file)
                    remaining = [s for s in all_responses if s["id"] not in evaluated_ids]

                    if not remaining:
                        print(f"All {total_expected} responses evaluated.")
                        break

                    print(f"Evaluating {len(remaining)}/{total_expected} responses using {config['num_workers']} workers...")

                    # Start vLLM server for local judge model (skipped when using OpenRouter)
                    if not eval_server and config["judge_server"] == 'local':
                        eval_server = start_vllm_server(eval_model, config["judge_tensor_parallel_size"], config["judge_data_parallel_size"], config)

                    if config["num_workers"] <= 1:
                        for sample in tqdm(remaining):
                            process_evaluation(sample, judge_file, eval_model, config["judge_server"], config)
                    else:
                        eval_worker_func = partial(process_evaluation, eval_file=judge_file, model_name=eval_model, judge_server=config["judge_server"], config=config)
                        with concurrent.futures.ProcessPoolExecutor(max_workers=config["num_workers"]) as executor:
                            futures = [executor.submit(eval_worker_func, sample) for sample in remaining]
                            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                                pass

                    # Re-check completeness after this pass
                    completed_count = len(load_completed_ids(judge_file))
                    if completed_count >= total_expected:
                        print(f"All {total_expected} responses evaluated.")
                        break
                    else:
                        print(f"Evaluated {completed_count}/{total_expected} responses. Retrying missing ones...")
                        # Responses whose judge reply cannot be parsed stay unevaluated, stop retrying them after a few passes
                        num_passes += 1
                        if completed_count - prev_completed <= int(0.02 * len(remaining)) or num_passes >= 10:
                            no_progress_passes += 3 if num_passes >= 10 else 1
                            if eval_server and config["judge_server"] == 'local':
                                kill_server_process(eval_server)
                                eval_server = None
                            if no_progress_passes >= 3:
                                print(f"Giving up on {total_expected - completed_count} responses that could not be evaluated.")
                                break
                        else:
                            no_progress_passes = 0
                    prev_completed = completed_count

            finally:
                if eval_server:
                    kill_server_process(eval_server)
                    eval_server = None

        # Merge the votes of all judges
        merge_judges(judge_files, eval_file)
        print("\nEvaluation completed.")

    # --- Phase 3: Aggregate scores ---
    if args.score:
        print("\nAggregating results...")
        eval_results = load_jsonl(eval_file)
        scores = {}

        # Initialize score accumulators per task
        for task in datasets_to_load:
            if 'mimic' in task:
                scores[task] = {'score_obtained': 0, 'score_total': 0}
            else:
                scores[task] = {'correct': 0, 'incorrect': 0}

        # Accumulate scores from eval results
        num_samples = {task: 0 for task in datasets_to_load}
        num_rule_scored = {task: 0 for task in datasets_to_load}
        for result in eval_results:
            task = "_".join(result.get("id").split('_')[:-1])
            score = result.get("evaluation")['score']
            num_samples[task] += 1
            if result.get("scored_by", "judge") != "judge":
                num_rule_scored[task] += 1
            if 'mimic' in result.get("id"):
                scores[task]['score_obtained'] += float(score)
                scores[task]['score_total'] += 5  # max score per sample
            else:
                if int(score):
                    scores[task]['correct'] += 1
                else:
                    scores[task]['incorrect'] += 1

        # Compute final accuracy per task, tasks without any scored sample are left out
        for task in datasets_to_load:
            if 'mimic' in task:
                obtained, total = scores[task]['score_obtained'], scores[task]['score_total']
            else:
                obtained, total = scores[task]['correct'], scores[task]['correct'] + scores[task]['incorrect']
            if total == 0:
                print(f"Warning: task '{task}' has no scored samples and is excluded from the results.")
                del scores[task]
            else:
                scores[task] = obtained / total

        # Build results table
        score_file = os.path.join(output_dir, f"{model_name_underscored}_score.txt")
        data = {'Task': list(scores.keys()), 'Accuracy': list(scores.values())}
        df = pd.DataFrame(data)
        df['Accuracy (%)'] = df['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
        df['N'] = [num_samples[task] for task in scores]
        df['Rule-scored (%)'] = [f"{100 * num_rule_scored[task] / num_samples[task]:.0f}%" for task in scores]

        # Macro average over tasks and micro average over samples
        if len(scores) > 1:
            total_samples = sum(num_samples[task] for task in scores)
            avg_score = sum(scores.values()) / len(scores)
            micro_score = sum(scores[task] * num_samples[task] for task in scores) / total_samples
            summary_row = pd.DataFrame({'Task': ['Overall', 'Overall_micro'],
                                    'Accuracy': [avg_score*100, micro_score*100],
                                    'Accuracy (%)': [f"{avg_score*100:.2f}%", f"{micro_score*100:.2f}%"],
                                    'N': [total_samples, total_samples],
                                    'Rule-scored (%)': [f"{100 * sum(num_rule_scored[task] for task in scores) / total_samples:.0f}%"] * 2})
            df = pd.concat([df, summary_row], ignore_index=True)

        print("\nEvaluation Results:")
        print(df.to_string(index=True))

        with open(score_file, 'a') as f:
            f.write(f"Model: {model_name}\n\n")
            f.write(df.to_string())

        print(f"\nResults saved to \033[1m{score_file}\033[0m\n")

if __name__ == "__main__":
    main()
