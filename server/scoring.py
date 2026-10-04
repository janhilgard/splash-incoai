"""Prefill-only decision scoring: POST /v1/score.

A score request renders its messages exactly as /v1/chat/completions does
(same chat template, reasoning effort, images and tokenization), appends
`assistant_prefix` after the assistant turn's opening, and reads the final
position's logits at its labels' single tokens in one native prefill, with
no decoding. The native result also carries the position's log-sum-exp over
the whole vocabulary and its highest logits, from which full-vocabulary log
probabilities, the labels' probability mass and the top tokens follow.

See docs/decision-scoring.md for the design and the API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Requests one batched /v1/score call may carry.
MAX_BATCH_REQUESTS = 64
# Top tokens a response may report: the native readout keeps this many.
MAX_TOP_K = 20

REQUEST_FIELDS = {
    "model",
    "messages",
    "assistant_prefix",
    "labels",
    "label_token_ids",
    "temperature",
    "return_top_k",
    "reasoning_effort",
    "chat_template_kwargs",
    "priority",
    "timeout",
}
BATCH_FIELDS = {"model", "requests", "timeout", "priority"}


class ScoreRequestError(ValueError):
    """A score request field is invalid; the message names it."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class ScoreOptions:
    """The validated scoring fields of one request (messages are validated by
    the chat preparation)."""

    assistant_prefix: str
    labels: tuple | None
    label_token_ids: tuple | None
    temperature: float
    return_top_k: int


def validate_options(body, *, index=None):
    """The scoring fields of one request body; raises ScoreRequestError."""
    where = "" if index is None else f"requests[{index}]."
    unknown = sorted(set(body) - REQUEST_FIELDS)
    if unknown:
        raise ScoreRequestError(f"unsupported fields: {', '.join(where + name for name in unknown)}")
    prefix = body.get("assistant_prefix", "")
    if prefix is None:
        prefix = ""
    if not isinstance(prefix, str):
        raise ScoreRequestError(f"{where}assistant_prefix must be a string")
    labels = body.get("labels")
    ids = body.get("label_token_ids")
    if ids is not None:
        if (
            not isinstance(ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids)
        ):
            raise ScoreRequestError(f"{where}label_token_ids must be an array of token ids")
        if len(ids) < 2:
            raise ScoreRequestError(f"{where}label_token_ids must name at least two tokens")
        if len(set(ids)) != len(ids):
            raise ScoreRequestError(f"{where}label_token_ids must be distinct")
        if labels is not None and (not isinstance(labels, list) or len(labels) != len(ids)):
            raise ScoreRequestError(
                f"{where}labels must match label_token_ids one to one when both are given"
            )
    if labels is not None:
        if not isinstance(labels, list) or any(not isinstance(label, str) or not label for label in labels):
            raise ScoreRequestError(f"{where}labels must be an array of nonempty strings")
    elif ids is None:
        raise ScoreRequestError(f"{where}labels or label_token_ids is required")
    if labels is not None and ids is None:
        if len(labels) < 2:
            raise ScoreRequestError(f"{where}labels must contain at least two labels")
        if len(set(labels)) != len(labels):
            raise ScoreRequestError(f"{where}labels must be distinct")
    temperature = body.get("temperature", 1.0)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ScoreRequestError(f"{where}temperature must be a positive number")
    top_k = body.get("return_top_k", 0)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= MAX_TOP_K:
        raise ScoreRequestError(f"{where}return_top_k must be an integer in [0, {MAX_TOP_K}]")
    return ScoreOptions(
        prefix,
        tuple(labels) if labels is not None else None,
        tuple(ids) if ids is not None else None,
        float(temperature),
        top_k,
    )


def label_breakdown(tokenizer, tokens):
    """Tokens with their text, for a 400 that shows why a label is not one
    token in its position."""
    return [{"id": token, "text": tokenizer.decode([token])} for token in tokens]


def resolve_labels(tokenizer, encode, text, ids, options, vocabulary):
    """The labels' token ids at the end of `text` (whose tokens are `ids`,
    as `encode` tokenizes it).

    A text label must be exactly one token in this position: the rendered
    text followed by the label must tokenize as `ids` plus that one token.
    Returns (labels, token_ids); raises ScoreRequestError with the label's
    token breakdown otherwise.
    """
    if options.label_token_ids is not None:
        token_ids = options.label_token_ids
        if any(token >= vocabulary for token in token_ids):
            raise ScoreRequestError("label_token_ids must be inside the vocabulary")
        labels = options.labels or tuple(tokenizer.decode([token]) for token in token_ids)
        return labels, token_ids
    token_ids = []
    problems = []
    for label in options.labels:
        extended = list(encode(text + label))
        if extended[: len(ids)] == list(ids) and len(extended) == len(ids) + 1:
            token_ids.append(extended[-1])
            continue
        tail = extended[len(ids):] if extended[: len(ids)] == list(ids) else None
        problems.append(
            {
                "label": label,
                "tokens": (
                    label_breakdown(tokenizer, tail)
                    if tail is not None
                    else "the label changes the tokenization of the prompt before it"
                ),
            }
        )
    if problems:
        raise ScoreRequestError(
            "each label must be exactly one token after the assistant prefix",
            {"labels": problems},
        )
    if len(set(token_ids)) != len(token_ids):
        raise ScoreRequestError("labels map to the same token")
    return tuple(options.labels), tuple(token_ids)


def log_softmax(values):
    maximum = max(values)
    total = sum(math.exp(value - maximum) for value in values)
    return [value - maximum - math.log(total) for value in values]


def score_response(model, labels, token_ids, options, result, *, tokenizer, cached_tokens, queue_ms):
    """The /v1/score result of one native score readout."""
    logits = [float(value) for value in result.option_logits]
    normalizer = float(result.log_normalizer)
    scaled = [value / options.temperature for value in logits]
    probabilities = [math.exp(value) for value in log_softmax(scaled)]
    full = [value - normalizer for value in logits]
    best = max(range(len(logits)), key=lambda index: (logits[index], -index))
    top = [
        {
            "token_id": int(token),
            "token": tokenizer.decode([int(token)]),
            "logprob": float(value) - normalizer,
        }
        for token, value in list(zip(result.top_token_ids, result.top_logits))[: options.return_top_k]
    ]
    return {
        "object": "score",
        "model": model,
        "labels": [
            {
                "label": label,
                "token_id": int(token),
                "logprob": logprob,
                "prob": probability,
                "logit": logit,
            }
            for label, token, logprob, probability, logit in zip(
                labels, token_ids, full, probabilities, logits
            )
        ],
        "label_mass": min(1.0, sum(math.exp(value) for value in full)),
        "argmax": labels[best],
        "top_k": top,
        "usage": {"prompt_tokens": result.prompt_tokens, "cached_tokens": cached_tokens},
        "timing": {
            "queue_ms": queue_ms,
            "prefill_ms": result.start_to_first_token_ms,
            "total_ms": result.request_wall_ms,
        },
    }
