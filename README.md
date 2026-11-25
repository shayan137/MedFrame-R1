<p align="center">
  <img src="assets/medframe_logo.png" width="420" alt="MedFrame logo">
</p>

# MedFrame-R1: A Framework for Reliable and Reproducible Medical Reinforcement Learning

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red.svg?logo=arXiv" alt="arXiv"></a>&nbsp;
  <a href="https://huggingface.co/collections/Shayannasiri/medframe-r1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-Hugging%20Face-yellow.svg" alt="Checkpoints"></a>&nbsp;
  <a href="DEMO_URL"><img src="https://img.shields.io/badge/Demo-Live-green.svg" alt="Demo"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg" alt="License"></a>
</p>

## Overview

MedFrame-R1 is an open-source framework for reinforcement learning (RL) of medical multimodal large language models (MLLMs) with rewards from an LLM judge. It keeps the standard group-based RL recipe and addresses four problems that make such pipelines unreliable:

- **Uninformative groups.** When all sampled answers of a question get the same score, the group gives no learning signal.
- **Reward exploitation.** A judge can be satisfied by listing many possible answers or by long, repetitive reasoning.
- **Missing domain knowledge.** RL cannot improve on questions that the starting model never answers correctly.
- **Evaluation bias.** A single judge decides both what is correct and which model is better.

---

## Components

- **Dynamic sampling on the judge decision.** Each step keeps only the groups whose rollouts received mixed judge decisions, and generation continues until the step is full. Spare groups can be carried over to the next step.
- **Embedding gate and length reward.** The judge reward only counts when the answer is also close to the reference in a medical embedding space, and a strict format check rejects malformed rollouts. The length reward rescales the advantages so that short correct answers are preferred and long incorrect answers are penalized.
- **Supervised cold start.** Before RL, the model is fine-tuned on chain-of-thought targets with an answer-weighted cross entropy and the Adaptive Early-Stop Loss (AESL).
- **Evaluation protocol.** Answers that can be decided in code are scored by fixed rules, and all remaining answers are scored by three judges from different model families with a majority vote.

---

## Checkpoints

The trained MedFrame-R1 checkpoints are released on Hugging Face: [huggingface.co/Shayannasiri](https://huggingface.co/Shayannasiri).

---

## Configuration

Every stage of a model is controlled by **one config file** in [`configs/`](configs):

| Config | Base model | Cold start | RL steps |
|--------|------------|------------|----------|
| `configs/qwen3.5_4b.yaml` (default) | Qwen3.5-4B | no | 100 |
| `configs/qwen3.5_9b.yaml` | Qwen3.5-9B | no | 100 |
| `configs/qwen3vl_4b.yaml` | Qwen3-VL-4B | wCE + AESL | 200 |
| `configs/qwen3vl_8b.yaml` | Qwen3-VL-8B | wCE + AESL | 200 |

| Section | Used by | Content |
|---------|---------|---------|
| `cold_start` | `training/run_cold_start.sh` | data, loss and optimization of the supervised cold start |
| `data`, `algorithm`, `worker`, `trainer` | `training/run_train.sh` | RL training, including dynamic sampling, the reward and the length reward |
| `eval` | `eval/eval.sh` | model to evaluate, judges and scoring rules |

---

## Training

The training pipeline uses a vLLM server for the judge reward during RL. All scripts take the config file as their only argument.

```bash
cd training
pip install -e .
bash run_cold_start.sh ../configs/qwen3vl_8b.yaml   # Step 0: Supervised cold start (Qwen3-VL only)
bash vllm_serve.sh ../configs/qwen3.5_4b.yaml       # Step 1: Start the reward server
bash run_train.sh ../configs/qwen3.5_4b.yaml        # Step 2: Launch RL training
bash merge_model.sh                                 # Step 3: Merge FSDP checkpoints
```

Training data: [MBZUAI/medix-rl-data](https://huggingface.co/datasets/MBZUAI/medix-rl-data) (~51K train, ~2.5K test samples)

See [`training/README.md`](training/README.md) for detailed setup and configuration options.

## Evaluation

The evaluation covers 17 medical benchmarks for both text-only (LLM) and image+text (VLM) tasks.

```bash
cd eval
pip install uv && uv pip install -r requirements.txt
bash eval.sh ../configs/qwen3.5_4b.yaml   # Run all phases: generate, evaluate, score
```

See [`eval/README.md`](eval/README.md) for the protocol, task selection and output files.


## Acknowledgements

We are thankful to [MediX-R1](https://github.com/mbzuai-oryx/MediX-R1) and [EasyR1](https://github.com/hiyouga/EasyR1) (a fork of [veRL](https://github.com/volcengine/verl)) for their open-source RL training framework and data, and to the authors of "Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT" for the AESL code.
