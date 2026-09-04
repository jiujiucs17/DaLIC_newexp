from copy import deepcopy
import json
from pathlib import Path

import litellm
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class ContextWindowModelConfig(LitellmModelConfig):
    context_length: int


def raise_template_exception(message: str) -> None:
    raise ValueError(message)


class ContextWindowModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=ContextWindowModelConfig, **kwargs)
        tokenizer_dir = Path(__file__).parent / "tokenizers"
        self._tokenizer = Tokenizer.from_file(
            str(tokenizer_dir / "qwen3_5_tokenizer.json")
        )
        tokenizer_config = json.loads(
            (tokenizer_dir / "qwen3_5_tokenizer_config.json").read_text()
        )
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = raise_template_exception
        environment.filters["tojson"] = lambda value: json.dumps(
            value, ensure_ascii=False
        )
        self._chat_template = environment.from_string(tokenizer_config["chat_template"])

    def _count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        render_messages = deepcopy(messages)
        for message in render_messages:
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments)

        rendered_prompt = self._chat_template.render(
            messages=render_messages,
            tools=[BASH_TOOL],
            add_generation_prompt=True,
            enable_thinking=True,
            add_vision_id=False,
        )
        return len(
            self._tokenizer.encode(rendered_prompt, add_special_tokens=False).ids
        )

    @staticmethod
    def _normalize_gpt_tool_response(response):
        """Merge a GPT tool-call choice into the primary choice."""
        choices = response.choices
        if not choices or choices[0].message.tool_calls:
            return response

        for choice in choices[1:]:
            if choice.message.tool_calls:
                choices[0].message.tool_calls = choice.message.tool_calls
                choices[0].finish_reason = choice.finish_reason
                response.choices = choices[:1]
                break
        return response

    def _query(self, messages: list[dict[str, str]], **kwargs):
        input_tokens = self._count_input_tokens(messages)
        remaining_tokens = self.config.context_length - input_tokens
        if remaining_tokens <= 0:
            raise litellm.exceptions.ContextWindowExceededError(
                message=(
                    f"Maximum context length is {self.config.context_length} tokens, "
                    f"but the prompt contains {input_tokens} tokens."
                ),
                model=self.config.model_name,
                llm_provider="openai",
            )

        is_gpt = self.config.model_name.rsplit("/", 1)[-1].lower().startswith("gpt")
        if is_gpt:
            kwargs["max_completion_tokens"] = remaining_tokens
        else:
            kwargs["max_tokens"] = remaining_tokens

        response = super()._query(messages, **kwargs)
        if is_gpt:
            response = self._normalize_gpt_tool_response(response)
        return response
