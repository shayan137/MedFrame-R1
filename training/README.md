# MedFrame-R1: Training Code

Training code for MedFrame-R1: a supervised cold start followed by RL with dynamic sampling, an embedding-gated judge reward and a length reward.

## Models

| Config | Base model | Cold start | Training judge |
|--------|------------|------------|----------------|
| `../configs/qwen3.5_4b.yaml` | `Qwen/Qwen3.5-4B` | no | `Qwen/Qwen3.5-9B` |
| `../configs/qwen3.5_9b.yaml` | `Qwen/Qwen3.5-9B` | no | `Qwen/Qwen3-14B` |
| `../configs/qwen3vl_4b.yaml` | `Qwen/Qwen3-VL-4B-Instruct` | wCE + AESL | `Qwen/Qwen3-8B` |
| `../configs/qwen3vl_8b.yaml` | `Qwen/Qwen3-VL-8B-Instruct` | wCE + AESL | `Qwen/Qwen3-14B` |

## Dataset

Training uses the [MBZUAI/medix-rl-data](https://huggingface.co/datasets/MBZUAI/medix-rl-data) dataset, which contains medical VQA samples with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `image` | Sequence[Image] | Medical image(s) for the question |
| `problem` | string | The medical question |
| `solution` | string | Ground truth answer |

The dataset has two splits:
- **train**: 51.3k samples
- **test**: 2.45k samples

The cold start additionally needs one reasoning target per training sample. Set `cold_start.thinking_file` to a JSONL file with one line per sample:

```
{"row_id": 0, "thinking": "The image shows ..."}
```

`row_id` is the index of the sample in the train split. Samples without a reasoning target are skipped.


## Installation

### Option 1: Docker

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

If your environment does not support Docker, you can use **Apptainer**:

```bash
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

### Option 2: Pip Install

```bash
cd MedFrame-R1/training
pip install -e .
```

### Software Requirements

- Python 3.9+
- transformers>=4.54.0
- flash-attn>=2.4.3
- vllm>=0.8.3

> **Note:** Qwen3.5 models need transformers>=5.2 and vllm>=0.17. The image above is sufficient for the Qwen3-VL configs only.


## Configuration

All settings are read from one config file. The sections used by training are:

| Section | Description |
|---------|-------------|
| `cold_start` | Supervised cold start: data, loss (`ce`, `wce`, `aesl`, `wce_aesl`) and optimization |
| `data` | RL data, prompt template, optional system prompt and batch sizes |
| `algorithm` | Dynamic sampling (`online_filtering`, `filter_key`), carry-over and the length reward |
| `worker.actor`, `worker.rollout` | Policy optimization and vLLM rollout settings |
| `worker.reward.reward_function_kwargs` | Judge model and URL, embedding gate, strict format check |
| `trainer` | Steps, validation, checkpoints and logging |

Any key can also be overridden on the command line, e.g. `python3 -m verl.trainer.main config=../configs/qwen3.5_4b.yaml trainer.max_steps=10`.

The reward of a rollout is

```
R = 0.1 * format + 0.9 * (0.575 * judge * gate + 0.375 * embedding + 0.05 * modality)
```

where `gate` is the embedding indicator when `embed_gate` is true and 1 otherwise. With `strict_format`, a rollout that violates the format gets `format_penalty` and is not judged. Dynamic sampling uses the judge decision before the gate (`filter_key: accuracy_llm`).


## Training

### Step 0: Supervised cold start (optional)

Configs with `cold_start.enable: true` first fine-tune the base model on reasoning targets:

```bash
bash run_cold_start.sh ../configs/qwen3vl_8b.yaml
```

Edit `run_cold_start.sh` to set the number of GPUs. The final model is saved to `<cold_start.output_dir>/final`, which is the `worker.actor.model.model_path` of RL in the same config.

### Step 1: Start the vLLM reward server

The LLM-based reward function calls a vLLM server to judge answer correctness. Start it **before** training:

```bash
bash vllm_serve.sh ../configs/qwen3.5_4b.yaml
```

The judge model and the port are read from `worker.reward.reward_function_kwargs` in the config file. Edit `vllm_serve.sh` to configure the GPU IDs and other server settings.

### Step 2: Start training

```bash
bash run_train.sh ../configs/qwen3.5_4b.yaml
```

Set `trainer.nnodes` and `trainer.n_gpus_per_node` to your hardware. Besides the checkpoints, the run directory contains:

- `experiment_log.jsonl`: metrics of each step, including `filter/*` (groups generated, trainable, easy and hard) and `length_reward/*`
- `sample_log.jsonl`: outcome of every sampled question in dynamic sampling
- `rollouts/step_*.jsonl.gz`: all rollouts with their rewards (if `trainer.log_rollouts` is true)

To train only on the samples that were trainable in a previous run, build an order file and set `data.order_file`:

```bash
python3 scripts/build_trainable_pool.py --sample_log checkpoints/<run>/sample_log.jsonl --output trainable_pool.jsonl
```

> **Note:** On AMD GPUs, set `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` before starting Ray so that each worker can select its device by local rank.

### Step 3: Merge checkpoint

After training, merge the FSDP sharded checkpoint into a HuggingFace-compatible model:

```bash
bash merge_model.sh
```

Edit `merge_model.sh` to point `--local_dir` to your checkpoint's actor directory.

## Acknowledgements

Built on top of [MediX-R1](https://github.com/mbzuai-oryx/MediX-R1) and [EasyR1](https://github.com/hiyouga/EasyR1), which is a fork of [veRL](https://github.com/volcengine/verl). The Adaptive Early-Stop Loss in `cold_start/aesl_loss.py` follows the official code of "Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT".
