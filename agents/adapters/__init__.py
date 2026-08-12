"""모델 어댑터 계층.

에이전트는 ModelAdapter 만 보고, 뒤에 로컬 모델이 붙었는지 외부 API가
붙었는지 알지 못한다. 설정으로 교체한다.
"""

from .base import (
    ChatMessage,
    ChatResponse,
    ImagePart,
    ModelAdapter,
    ToolCall,
    ToolSpec,
    parse_json_object,
)
from .config import (
    DEFAULT_LOCAL_BASE_URL,
    ModelConfig,
    RuntimeConfig,
    build_adapter,
    build_adapters,
    load_config,
)
from .openai_compat import OpenAICompatAdapter
from .stub import StubAdapter

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "ImagePart",
    "ModelAdapter",
    "ToolCall",
    "ToolSpec",
    "parse_json_object",
    "DEFAULT_LOCAL_BASE_URL",
    "ModelConfig",
    "RuntimeConfig",
    "build_adapter",
    "build_adapters",
    "load_config",
    "OpenAICompatAdapter",
    "StubAdapter",
]
