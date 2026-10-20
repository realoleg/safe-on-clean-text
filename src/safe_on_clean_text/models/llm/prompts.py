from __future__ import annotations


def build_messages(llm_config: dict, prompt_id: str, method: str, text: str) -> list[dict[str, str]]:
    """
    Chat messages of one example: system message, then the prompt. For S-verb the verbal instruction
    replaces the prompt's answer-format sentence (pre-run correction: one answer format per prompt).
    """

    user = llm_config["prompts"][prompt_id]
    if method == "s_verb":
        answer_format = llm_config["answer_format"][prompt_id]
        assert user.endswith(answer_format), prompt_id
        user = user[: -len(answer_format)] + llm_config["verbal_instruction"]
    user = user.replace("{text}", text)
    return [{"role": "system", "content": llm_config["system"]}, {"role": "user", "content": user}]
