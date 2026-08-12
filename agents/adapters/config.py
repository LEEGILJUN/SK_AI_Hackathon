"""어댑터를 설정으로 만든다.

언어 모델과 시각 언어 모델을 따로 설정한다. 같은 모델을 쓸 수도 있지만,
4090 한 대에서 돌릴 때는 텍스트용 작은 모델과 시각용 모델을 나누는 편이
메모리 운용에 유리해서 처음부터 분리해 둔다.

우선순위는 환경 변수 > YAML > 기본값이다. 기본값은 스텁이라, 아무 설정도
없으면 모델 없이도 흐름은 돌아간다. 다만 판정은 비어 있다.

환경 변수 (VLM 은 SHVO_VLM_* 로 같은 형태)

    SHVO_LLM_PROVIDER   stub | openai_compat
    SHVO_LLM_BASE_URL   http://localhost:11434/v1
    SHVO_LLM_MODEL      모델 이름
    SHVO_LLM_API_KEY    외부 API 를 쓸 때만
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

from .base import ModelAdapter
from .openai_compat import OpenAICompatAdapter
from .stub import StubAdapter

Provider = Literal["stub", "openai_compat"]

#: 로컬 실행의 기본 후보. Ollama 가 이 주소를 쓴다.
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"


@dataclass
class ModelConfig:
    provider: Provider = "stub"
    model: str = ""
    base_url: str | None = None
    api_key_env: str = ""
    timeout: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeConfig:
    """에이전트 실행에 필요한 모델 설정 묶음."""

    llm: ModelConfig
    vlm: ModelConfig

    def to_dict(self) -> dict[str, Any]:
        return {"llm": self.llm.to_dict(), "vlm": self.vlm.to_dict()}


def _from_env(prefix: str, base: ModelConfig) -> ModelConfig:
    """환경 변수로 덮어쓴다. 값이 없는 항목은 그대로 둔다."""

    def pick(key: str, current):
        value = os.environ.get(f"{prefix}_{key}")
        return value if value not in (None, "") else current

    provider = pick("PROVIDER", base.provider)
    if provider not in ("stub", "openai_compat"):
        raise ValueError(
            f"{prefix}_PROVIDER 값이 잘못됐다: {provider!r}. stub 또는 openai_compat 이어야 한다."
        )

    base_url = pick("BASE_URL", base.base_url)
    if provider == "openai_compat" and not base_url:
        base_url = DEFAULT_LOCAL_BASE_URL

    return ModelConfig(
        provider=provider,  # type: ignore[arg-type]
        model=pick("MODEL", base.model),
        base_url=base_url,
        api_key_env=f"{prefix}_API_KEY",
        timeout=float(pick("TIMEOUT", base.timeout)),
        temperature=float(pick("TEMPERATURE", base.temperature)),
        max_tokens=int(pick("MAX_TOKENS", base.max_tokens)),
        seed=base.seed,
    )


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    """YAML 과 환경 변수를 합쳐 설정을 만든다.

    path 가 없거나 파일이 없으면 기본값(스텁)에서 시작한다. 저장소에 설정
    파일을 커밋하지 않아도 각자 환경 변수로 돌릴 수 있게 하기 위함이다.
    """
    llm = ModelConfig()
    vlm = ModelConfig()

    if path:
        file_path = Path(path)
        if file_path.exists():
            import yaml

            data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
            if "llm" in data:
                llm = ModelConfig(**{**asdict(llm), **data["llm"]})
            if "vlm" in data:
                vlm = ModelConfig(**{**asdict(vlm), **data["vlm"]})

    return RuntimeConfig(llm=_from_env("SHVO_LLM", llm), vlm=_from_env("SHVO_VLM", vlm))


def build_adapter(config: ModelConfig) -> ModelAdapter:
    """설정 하나로 어댑터를 만든다."""
    if config.provider == "stub":
        return StubAdapter()

    if not config.model:
        raise ValueError(
            "openai_compat 을 쓰려면 모델 이름이 필요하다. "
            "SHVO_LLM_MODEL 또는 SHVO_VLM_MODEL 을 설정하라."
        )

    return OpenAICompatAdapter(
        model=config.model,
        base_url=config.base_url,
        api_key=os.environ.get(config.api_key_env) if config.api_key_env else None,
        timeout=config.timeout,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )


def build_adapters(path: str | Path | None = None) -> tuple[ModelAdapter, ModelAdapter]:
    """(언어 모델, 시각 언어 모델) 어댑터 쌍을 만든다."""
    config = load_config(path)
    return build_adapter(config.llm), build_adapter(config.vlm)
