"""Model registry for Agent runtime roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.core.config import ServiceType, settings
from app.core.logger import get_logger

logger = get_logger(service="harness.models")


@dataclass(frozen=True)
class ModelProfile:
    """Runtime model profile selected for a specific role."""

    role: str
    provider: str
    model: str
    temperature: float = 0.7


class ModelRegistry:
    """Creates chat model instances from role-based configuration.

    Roles intentionally map to different configuration knobs so the project can
    use separate models for normal chat, reasoning, agent orchestration, and
    vision preprocessing.
    """

    def get_chat_model(self, *, tags: Sequence[str] | None = None) -> BaseChatModel:
        return self._build_text_model(
            role="chat",
            service=settings.CHAT_SERVICE,
            ollama_model=settings.OLLAMA_CHAT_MODEL,
            tags=tags,
        )

    def get_reason_model(self, *, tags: Sequence[str] | None = None) -> BaseChatModel:
        return self._build_text_model(
            role="reason",
            service=settings.REASON_SERVICE,
            ollama_model=settings.OLLAMA_REASON_MODEL,
            tags=tags,
        )

    def get_agent_model(self, *, tags: Sequence[str] | None = None) -> BaseChatModel:
        return self._build_text_model(
            role="agent",
            service=settings.AGENT_SERVICE,
            ollama_model=settings.OLLAMA_AGENT_MODEL,
            tags=tags,
        )

    def get_vision_profile(self) -> ModelProfile:
        return ModelProfile(
            role="vision",
            provider="openai-compatible",
            model=settings.VISION_MODEL,
        )

    def describe_agent_model(self) -> ModelProfile:
        if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
            return ModelProfile(role="agent", provider="deepseek", model=settings.DEEPSEEK_MODEL)
        return ModelProfile(role="agent", provider="ollama", model=settings.OLLAMA_AGENT_MODEL)

    def _build_text_model(
        self,
        *,
        role: str,
        service: ServiceType,
        ollama_model: str,
        tags: Sequence[str] | None,
    ) -> BaseChatModel:
        normalized_tags = list(tags or [])
        if service == ServiceType.DEEPSEEK:
            logger.info(
                f"Using OpenAI-compatible model {settings.DEEPSEEK_MODEL} "
                f"for {role} via {settings.DEEPSEEK_BASE_URL}"
            )
            return ChatOpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                model=settings.DEEPSEEK_MODEL,
                temperature=0.7,
                tags=normalized_tags,
            )

        logger.info(f"Using Ollama model {ollama_model} for {role}")
        return ChatOllama(
            model=ollama_model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.7,
            tags=normalized_tags,
        )
