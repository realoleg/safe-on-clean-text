from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from safe_on_clean_text.models.llm.confidence import label_probabilities, parse_verbal, parse_yes_no


class LocalModel:
    """
    A 4-bit MLX model, loaded on first use. S-tok reads the next-token distribution after the chat template
    (thinking disabled; optional answer prefix in the assistant turn); S-verb decodes greedily.
    """

    def __init__(self, repo: str, llm_config: dict) -> None:
        self.repo = repo
        self.llm_config = llm_config
        self.model = None
        self.tokenizer = None
        self.calls = 0

    def _load(self) -> None:
        if self.model is not None:
            return
        from mlx_lm import load

        self.model, self.tokenizer = load(self.repo)
        self.label_ids = {name: self._single_token_ids(self.llm_config[f"{name}_tokens"]) for name in ["yes", "no"]}

    def _single_token_ids(self, variants: list[str]) -> list[int]:
        ids = [self.tokenizer.encode(v, add_special_tokens=False) for v in variants]
        return sorted({tokens[0] for tokens in ids if len(tokens) == 1})

    def prompt_ids(self, messages: list[dict[str, str]], answer_prefix: str) -> list[int]:
        self._load()
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        return self.tokenizer.encode(prompt + answer_prefix, add_special_tokens=False)

    def s_tok(self, messages: list[dict[str, str]], answer_prefix: str) -> dict[str, Any]:
        import mlx.core as mx

        ids = self.prompt_ids(messages, answer_prefix)
        # Softmax in float64 so that probabilities near 0 / 1 keep their tails (float32 rounds p to exactly 1).
        logits = np.array(self.model(mx.array(ids)[None])[0, -1].astype(mx.float32)).astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        self.calls += 1
        return {
            **label_probabilities(float(probs[self.label_ids["yes"]].sum()), float(probs[self.label_ids["no"]].sum())),
            "top_token_id": int(probs.argmax()),
            "n_prompt_tokens": len(ids),
        }

    def s_verb(self, messages: list[dict[str, str]], answer_prefix: str) -> dict[str, Any]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        ids = self.prompt_ids(messages, answer_prefix)
        answer = generate(
            self.model,
            self.tokenizer,
            ids,
            max_tokens=int(self.llm_config["verbal_max_tokens"]),
            sampler=make_sampler(temp=0.0),
        )
        self.calls += 1
        return {"answer": answer, "p_stress": parse_verbal(answer), "n_prompt_tokens": len(ids)}


class RateLimiter:
    """
    Smooth client-side limit (thread-safe): request starts are spaced by max(60 / RPM, 60 * tokens / TPM) seconds,
    with RPM = min(request cap, fraction x account RPM) and TPM = fraction x account TPM. The account limits come from
    the x-ratelimit-limit-* headers of every response; until the first one only the request cap applies.
    """

    def __init__(self, max_requests_per_minute: float, fraction: float) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.fraction = fraction
        self.requests_per_minute = max_requests_per_minute
        self.tokens_per_minute = math.inf
        self.account_limits: dict[str, int] = {}
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self, tokens: int) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + max(60 / self.requests_per_minute, 60 * tokens / self.tokens_per_minute)
        time.sleep(max(0.0, start - now))

    def update(self, headers: Any) -> None:
        limits = {
            name: int(headers[f"x-ratelimit-limit-{name}"])
            for name in ["requests", "tokens"]
            if headers.get(f"x-ratelimit-limit-{name}")
        }
        with self._lock:
            if limits == self.account_limits:
                return
            self.account_limits = limits
            if "requests" in limits:
                self.requests_per_minute = min(self.max_requests_per_minute, self.fraction * limits["requests"])
            if "tokens" in limits:
                self.tokens_per_minute = self.fraction * limits["tokens"]
        print(
            f"Account rate limits {limits} -> client limits {self.requests_per_minute:.0f} RPM, "
            f"{self.tokens_per_minute:.0f} TPM",
            flush=True,
        )


class APIModel:
    """
    OpenAI Chat Completions on a dated snapshot. S-tok: first-token top_logprobs; S-verb: greedy short answer;
    S-cons: n samples at the consistency temperature in one request. Retries with exponential backoff
    are done by the client; calls and token usage are counted (thread-safe).
    """

    def __init__(self, model: str, llm_config: dict) -> None:
        from openai import OpenAI

        self.model = model
        self.llm_config = llm_config
        self.client = OpenAI(max_retries=int(llm_config["api_max_retries"]))
        self.limiter = RateLimiter(
            float(llm_config["api_max_requests_per_minute"]), float(llm_config["api_rate_limit_fraction"])
        )
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def _create(self, messages: list[dict[str, str]], **kwargs) -> Any:
        # Tokens counted against TPM: prompt (about 4 characters per token, plus chat overhead) + requested completions.
        estimate = (
            sum(len(m["content"]) for m in messages) // 4 + 20 + kwargs["max_completion_tokens"] * kwargs.get("n", 1)
        )
        self.limiter.wait(estimate)
        raw = self.client.chat.completions.with_raw_response.create(model=self.model, messages=messages, **kwargs)
        self.limiter.update(raw.headers)
        response = raw.parse()
        with self._lock:
            self.calls += 1
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
        return response

    def s_tok(self, messages: list[dict[str, str]], answer_prefix: str) -> dict[str, Any]:
        assert answer_prefix == "", "The OpenAI API cannot prefill the assistant turn."
        response = self._create(
            messages,
            temperature=0,
            max_completion_tokens=1,
            logprobs=True,
            top_logprobs=int(self.llm_config["top_logprobs"]),
        )
        top = response.choices[0].logprobs.content[0].top_logprobs
        logprob = {entry.token: entry.logprob for entry in top}
        # A variant absent from the top list gets the smallest returned logprob (an upper bound; pre-run correction).
        floor = min(logprob.values())
        out: dict[str, Any] = {}
        for name in ["yes", "no"]:
            variants = self.llm_config[f"{name}_tokens"]
            out[f"p_{name}_observed"] = sum(math.exp(logprob[v]) for v in variants if v in logprob)
            out[f"n_missing_{name}"] = sum(v not in logprob for v in variants)
        p_yes = out["p_yes_observed"] + out["n_missing_yes"] * math.exp(floor)
        p_no = out["p_no_observed"] + out["n_missing_no"] * math.exp(floor)
        return {
            **label_probabilities(p_yes, p_no),
            # The mass check uses the observed variants only.
            "mass": out["p_yes_observed"] + out["p_no_observed"],
            **out,
            "logprob_floor": floor,
            **usage(response),
        }

    def s_verb(self, messages: list[dict[str, str]], answer_prefix: str) -> dict[str, Any]:
        assert answer_prefix == ""
        response = self._create(
            messages, temperature=0, max_completion_tokens=int(self.llm_config["verbal_max_tokens"])
        )
        answer = response.choices[0].message.content or ""
        return {"answer": answer, "p_stress": parse_verbal(answer), **usage(response)}

    def s_cons(self, messages: list[dict[str, str]], answer_prefix: str) -> dict[str, Any]:
        assert answer_prefix == ""
        settings = self.llm_config["consistency"]
        response = self._create(
            messages,
            n=int(settings["n"]),
            temperature=float(settings["temperature"]),
            max_completion_tokens=int(settings["max_tokens"]),
        )
        answers = [choice.message.content or "" for choice in response.choices]
        votes = [vote for vote in map(parse_yes_no, answers) if vote is not None]
        return {
            "answers": answers,
            "n_recognized": len(votes),
            # All answers unrecognized -> 0.5 with a flag.
            "p_stress": float(np.mean(votes)) if votes else 0.5,
            "all_unrecognized": not votes,
            **usage(response),
        }


def usage(response: Any) -> dict[str, Any]:
    """
    Token usage plus the snapshot that answered and its system fingerprint.
    """

    return {
        "n_prompt_tokens": response.usage.prompt_tokens,
        "n_completion_tokens": response.usage.completion_tokens,
        "response_model": response.model,
        "system_fingerprint": response.system_fingerprint,
    }
