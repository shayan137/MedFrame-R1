import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util


# Metadata
REWARD_NAME = "medical"
REWARD_TYPE = "batch"

# Reward component weights (must sum to 1.0 when combined with format_weight)
ACCURACY_LLM_WEIGHT = 0.575
ACCURACY_EMBED_WEIGHT = 0.375
MODALITY_WEIGHT = 0.05

# Symbols that indicate a garbage answer
INVALID_SYMBOLS = ["__", ":", "?"]

# Exactly one thinking block and one answer block, with no tag inside either block
STRICT_FORMAT_PATTERN = re.compile(
    r"<thinking>(?:(?!</?thinking>|</?answer>).)*</thinking>\s*"
    r"<answer>(?:(?!</?thinking>|</?answer>).)*</answer>",
    re.DOTALL,
)

# First standalone YES or NO in the judge reply, ignoring an echoed "YES/NO"
VERDICT_PATTERN = re.compile(r"\b(YES|NO)\b")
LABEL_PATTERN = re.compile(r"\b(?:YES\s*(?:/|\s+OR\s+)\s*NO|NO\s*(?:/|\s+OR\s+)\s*YES|Y\s*/\s*N)\b")

# Loaded on first use inside the reward worker
embed_models = {}
judge_clients = {}


def format_reward(predict: str, strict: bool = False) -> float:
    """Checks if the response follows the <thinking>...</thinking><answer>...</answer> format."""
    idx = predict.find("<thinking>")
    if idx == -1:
        return 0.0

    predict_new = predict[idx:].strip()
    if strict:
        pattern = STRICT_FORMAT_PATTERN
    else:
        pattern = re.compile(r"<thinking>.*?</thinking>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_new)
    return 1.0 if format_match else 0.0


def create_prompt(predict: str, ground_truth: str) -> str:
    return f"""You are a helpful assistant.
Given a predicted answer and the ground truth answer, respond with exactly "YES" if the content matches, otherwise respond with "NO".
Start your response with "YES" or "NO" in uppercase.


Predicted:{predict}
Ground Truth:{ground_truth}
"""


def chat_with_vllm(prompt, model="Qwen/Qwen3-4B-Instruct-2507", url="http://localhost:8085/v1"):
    """Send a prompt to a vLLM server and return the response text."""
    if url not in judge_clients:
        judge_clients[url] = OpenAI(base_url=url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

    completion = judge_clients[url].chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (completion.choices[0].message.content or "").strip()


def llm_answer_match(predict: str, ground_truth: str, model: str, url: str) -> float:
    """Use vLLM chat API to ask yes/no if answers match."""
    prompt = create_prompt(predict, ground_truth)
    try:
        reply = chat_with_vllm(prompt, model, url)
        verdict = VERDICT_PATTERN.search(LABEL_PATTERN.sub(" ", reply.upper()))
        return 1.0 if verdict and verdict.group(1) == "YES" else 0.0
    except Exception as e:
        print(f"LLM API error: {e}")
        return 0.0


def _extract_answer(predict: str, ground_truth: str) -> tuple[str, str]:
    """Extract the predicted answer from <answer> tags and clean up ground truth."""
    content_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    given_answer = content_match.group(1).strip() if content_match else predict.strip()
    given_answer = given_answer.strip(".")

    ground_truth = ground_truth.split(">", maxsplit=1)[1].strip()
    ground_truth = ground_truth.strip(".")

    return given_answer, ground_truth


def _is_invalid_answer(answer: str) -> bool:
    """Check if an answer is empty, too short, or contains invalid symbols."""
    if answer == "" or len(answer) == 1:
        return True
    return any(symbol in answer for symbol in INVALID_SYMBOLS)


def embedding_similarity(answer_pairs: list[tuple[str, str]], model: str, batch_size: int = 16) -> list[float]:
    """Compute the embedding cosine similarity of (answer, ground truth) pairs in one encoder pass."""
    if model not in embed_models:
        embed_models[model] = SentenceTransformer(model, trust_remote_code=True)

    texts = [text for answer_pair in answer_pairs for text in answer_pair]
    embeddings = embed_models[model].encode(
        texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False
    )
    return [util.pytorch_cos_sim(embeddings[2 * i], embeddings[2 * i + 1]).item() for i in range(len(answer_pairs))]


def modality_reward(predict: str, ground_truth: str) -> float:
    """Reward for correctly predicting the image modality (before the <thinking> tag)."""
    idx = predict.find("<thinking>")
    if idx == -1:
        return 0.0

    predict_modality = predict[:idx].strip()
    ground_truth_modality = ground_truth.split(">", maxsplit=1)[0] + ">"

    return 1.0 if predict_modality.upper() == ground_truth_modality.upper() else 0.0


def compute_score(
    reward_inputs: list[dict[str, str]],
    format_weight: float = 0.1,
    judge_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    judge_url: str = "http://localhost:8085/v1",
    judge_workers: int = 16,
    embed_model: str = "abhinand/MedEmbed-large-v0.1",
    embed_batch_size: int = 16,
    embed_threshold: float = 0.8,
    embed_gate: bool = False,
    strict_format: bool = False,
    format_penalty: float = -1.0,
    strip_think: bool = False,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for medical reward function.")

    content_weight = 1 - format_weight
    scores, judged_scores, answer_pairs = [], [], []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        if strip_think:
            # Score the text after the think block only
            response = response.rsplit("</think>", 1)[1] if "</think>" in response else ""

        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
        ground_truth = reward_input["ground_truth"]

        score = {
            "overall": 0.0,
            "format": format_reward(predict, strict=strict_format),
            "accuracy_llm": 0.0,
            "accuracy_embed": 0.0,
            "modality": modality_reward(predict, ground_truth),
            "format_violation": 0.0,
            "embed_sim": 0.0,
            "gate_fired": 0.0,
        }
        scores.append(score)

        # A format violation gets the penalty and is not judged
        if strict_format and score["format"] == 0.0:
            score["format_violation"] = 1.0
            continue

        try:
            given_answer, ground_truth = _extract_answer(predict, ground_truth)
        except Exception:
            continue

        if _is_invalid_answer(given_answer):
            continue

        if given_answer == ground_truth:
            score["accuracy_llm"] = score["accuracy_embed"] = score["embed_sim"] = 1.0
            continue

        judged_scores.append(score)
        answer_pairs.append((given_answer, ground_truth))

    if answer_pairs:
        with ThreadPoolExecutor(max_workers=min(max(judge_workers, 1), len(answer_pairs))) as pool:
            futures = [
                pool.submit(llm_answer_match, given_answer, ground_truth, judge_model, judge_url)
                for given_answer, ground_truth in answer_pairs
            ]
            # Embeddings are computed while the judge requests are running
            embed_sims = embedding_similarity(answer_pairs, embed_model, embed_batch_size)
            for score, future, embed_sim in zip(judged_scores, futures, embed_sims):
                score["accuracy_llm"] = future.result()
                score["accuracy_embed"] = float(embed_sim >= embed_threshold)
                score["embed_sim"] = embed_sim

    for score in scores:
        if score["format_violation"]:
            score["overall"] = format_penalty
            continue

        # With the gate, the judge only counts when the embedding agrees
        gate = score["accuracy_embed"] if embed_gate else 1.0
        score["gate_fired"] = float(score["accuracy_llm"] > 0.0 and gate < 1.0)
        score["overall"] = (
            content_weight * ACCURACY_LLM_WEIGHT * score["accuracy_llm"] * gate
            + content_weight * ACCURACY_EMBED_WEIGHT * score["accuracy_embed"]
            + content_weight * MODALITY_WEIGHT * score["modality"]
            + format_weight * score["format"]
        )

    return scores
