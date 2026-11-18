# MedFrame-R1 Evaluation

Evaluation pipeline for medical LLM and VLM models. Supports response generation via vLLM, rule-based scoring, evaluation with several LLM judges, and score aggregation.

## Protocol

1. **Generate.** The model answers every sample once with greedy decoding.
2. **Rules.** Answers that can be decided in code are scored by fixed rules: multiple-choice questions by the selected option, yes/no questions by the stated verdict, and one-word references by a whole-word match. A thinking block is removed before scoring.
3. **Judges.** All remaining answers are scored by each judge in `eval.judges`. Binary tasks use the majority vote of the judges, and the MIMIC-CXR report tasks use the median of the 0-5 scores.
4. **Score.** Per-task accuracy, the macro average over tasks (`Overall`) and the micro average over samples (`Overall_micro`).

## Setup

### 1. Create a conda environment

```bash
conda create -n medframe_eval python=3.11 -y
conda activate medframe_eval
```

### 2. Install uv and dependencies

```bash
pip install uv
uv pip install -r requirements.txt
```

### 3. Configure credentials (MIMIC-CXR only)

If you plan to evaluate on MIMIC-CXR tasks (`mimic_cxr_report_generation`, `mimic_cxr_report_summarization`), create a `.env` file in the `eval/` directory with your [PhysioNet](https://physionet.org/) credentials:

```
PHYSIONET_UNAME=your_username
PHYSIONET_PWD=your_password
```

This is not required for any other tasks and can be skipped.

## Running the Evaluation

All settings are read from the `eval` section of the model config file. The pipeline has three phases that can be run independently or together: **generate**, **evaluate**, and **score**.

### Quick start (all phases at once)

```bash
bash eval.sh ../configs/qwen3.5_4b.yaml
```

### Running phases individually

#### Step 1: Generate responses

Starts a vLLM server with the candidate model, generates responses for each sample, and saves them to a JSONL file. Supports resuming -- already processed sample IDs are skipped automatically.

```bash
python eval.py --config ../configs/qwen3.5_4b.yaml --generate true
```

#### Step 2: Evaluate responses

Scores each response with the fixed rules or with the judges. Starts a vLLM server for one judge at a time, then merges the votes. Also supports resuming.

```bash
python eval.py --config ../configs/qwen3.5_4b.yaml --evaluate true
```

**Alternative: Use OpenRouter instead of self-hosting the judge models**

Set `judge_server: openrouter` in the config, use OpenRouter model names in `judges`, and export your credentials before running:

```bash
export OPENROUTER_API_KEY=your_openrouter_key
```

#### Step 3: Aggregate scores

Computes per-task accuracy and prints a summary table. No GPU required.

```bash
python eval.py --config ../configs/qwen3.5_4b.yaml --score true
```

## Config Reference

| Key | Default | Description |
|---|---|---|
| `model` | *(required)* | Candidate model for generation |
| `name` | model name | Name of the output folder and files |
| `tasks` | `["all"]` | Tasks to run (see task selection below) |
| `tasks_file` | `tasks.yaml` | Path to dataset config file |
| `output_dir` | `results` | Output directory |
| `num_workers` | `128` | Parallel workers for generation/evaluation |
| `system_prompt` | `null` | Text file used as the system prompt of the candidate model |
| `max_tokens` | `2048` | Maximum length of a response |
| `request_timeout` | `180` | Timeout of a generation request in seconds |
| `tensor_parallel_size`, `data_parallel_size` | `1`, `1` | vLLM parallelism of the candidate model |
| `max_model_len` | `32768` | Context window of the vLLM servers |
| `gpu_memory_utilization` | `0.95` | GPU memory fraction of the vLLM servers |
| `enforce_eager`, `seed` | `true`, `0` | vLLM settings for reproducible generation |
| `judge_server` | `local` | Judge server: `local` (vLLM) or `openrouter` |
| `judges` | one Qwen3-14B judge | List of judges, each with a `name` and a `model` |
| `judge_tensor_parallel_size`, `judge_data_parallel_size` | `1`, `1` | vLLM parallelism of the judges |
| `judge_rounds` | `1` | Independent votes of each judge per sample |
| `judge_max_tokens`, `judge_max_tokens_mimic` | `4096`, `16384` | Token budget of a judge reply |
| `judge_timeout` | `600` | Timeout of a judge request in seconds |
| `exact_max_words` | `25` | Longest response matched against a one-word reference by rule |
| `binary_scope_words` | `30` | Opening words of a response read for a yes/no verdict |
| `binary_final_words` | `12` | Words read after a final answer marker for a yes/no verdict |

## Task Selection

Tasks are defined in `tasks.yaml` under two categories: `llm` and `vlm`.

| Selector | Description |
|---|---|
| `all` | Run all tasks |
| `llm` | Run all text-only tasks |
| `vlm` | Run all vision-language tasks |
| `<task_name>` | Run a specific task (e.g. `medqa`, `slake_vqa`) |

Multiple tasks can be specified: `tasks: ["medqa", "slake_vqa", "pmc_vqa"]`

### Available tasks

**LLM** (text-only): `mmlu_clinical_knowledge`, `mmlu_college_biology`, `mmlu_college_medicine`, `mmlu_medical_genetics`, `mmlu_professional_medicine`, `mmlu_anatomy`, `medmcqa`, `medqa`, `usmle_sa`, `pubmedqa`, `mimic_cxr_report_summarization`

**VLM** (vision-language): `slake_vqa`, `rad_vqa`, `path_vqa`, `pmc_vqa`, `pmc_vqa_hard`, `mimic_cxr_report_generation`

## Output Structure

```
results/
  <name>/
    <name>.jsonl                  # Generated responses
    <name>_eval_<judge>.jsonl     # Evaluation results of one judge
    <name>_eval.jsonl             # Merged evaluation results
    <name>_score.txt              # Final score table
```

Each evaluation result has a `scored_by` field: `mcq_letter`, `mcq_text`, `binary` and `exact_short` for the fixed rules, and `judge` otherwise. Merged results also keep the vote of every judge in `judge_votes`.
