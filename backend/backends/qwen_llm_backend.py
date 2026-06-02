"""
Qwen3 LLM backend — MLX on Apple Silicon only.
"""

import asyncio
import logging
from typing import Optional

from . import LLMBackend, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE
from .base import is_model_cached, model_load_progress
from ..utils.hf_offline_patch import force_offline_if_cached

logger = logging.getLogger(__name__)


MLX_HF_REPOS = {
    "0.6B": "mlx-community/Qwen3-0.6B-4bit",
    "1.7B": "mlx-community/Qwen3-1.7B-4bit",
    "4B": "mlx-community/Qwen3-4B-4bit",
}


def _progress_name(model_size: str) -> str:
    return f"qwen3-{model_size.lower()}"


def _build_messages(
    prompt: str,
    system: Optional[str],
    examples: Optional[list[tuple[str, str]]] = None,
) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if examples:
        for user_text, assistant_text in examples:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": prompt})
    return messages


class MLXQwenLLMBackend:
    """Qwen3 LLM backend using mlx-lm (Apple Silicon)."""

    def __init__(self, model_size: str = "0.6B"):
        self.model = None
        self.tokenizer = None
        self.model_size = model_size
        self._current_model_size: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in MLX_HF_REPOS:
            raise ValueError(f"Unknown Qwen3 size: {model_size}")
        return MLX_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model(self, model_size: Optional[str] = None) -> None:
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        await asyncio.to_thread(self._load_model_sync, model_size)

    def _load_model_sync(self, model_size: str) -> None:
        from mlx_lm import load as mlx_load

        progress_model_name = _progress_name(model_size)
        is_cached = self._is_model_cached(model_size)
        repo = self._get_model_path(model_size)

        with model_load_progress(progress_model_name, is_cached):
            logger.info("Loading Qwen3 %s via MLX...", model_size)
            with force_offline_if_cached(is_cached, progress_model_name):
                loaded = mlx_load(repo)

        # mlx_lm.load returns (model, tokenizer) by default and
        # (model, tokenizer, config) when return_config=True.
        self.model = loaded[0]
        self.tokenizer = loaded[1]

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen3 %s (MLX) loaded successfully", model_size)

    def unload_model(self) -> None:
        if self.model is None:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._current_model_size = None
        logger.info("Qwen3 (MLX) unloaded")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        await self.load_model(model_size)
        return await asyncio.to_thread(
            self._generate_sync, prompt, system, max_tokens, temperature, examples
        )

    def _generate_sync(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        messages = _build_messages(prompt, system, examples)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        sampler = make_sampler(temp=temperature, top_p=0.9) if temperature > 0 else None
        text = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text.strip()
