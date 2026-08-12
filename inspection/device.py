"""실행 장치 선택.

개발은 Apple Silicon 노트북(mps), 시연은 RTX 4090 노트북(cuda)에서 돌아간다.
같은 코드가 양쪽에서 그대로 동작해야 하므로 장치 선택을 한곳에 모은다.
"""

from __future__ import annotations

import torch

_PREFERENCE = ("cuda", "mps", "cpu")


def available_devices() -> list[str]:
    """지금 이 머신에서 쓸 수 있는 장치 목록을 우선순위 순으로 돌려준다."""
    found = ["cpu"]
    if torch.backends.mps.is_available():
        found.insert(0, "mps")
    if torch.cuda.is_available():
        found.insert(0, "cuda")
    return found


def pick_device(prefer: str | None = None) -> torch.device:
    """쓸 장치를 고른다.

    prefer 를 주면 그것을 우선하되, 쓸 수 없으면 조용히 다음 순위로 내려간다.
    시연 장비가 바뀌어도 스크립트를 고치지 않게 하기 위함이다.
    """
    usable = available_devices()
    if prefer and prefer in usable:
        return torch.device(prefer)
    for name in _PREFERENCE:
        if name in usable:
            return torch.device(name)
    return torch.device("cpu")


def describe(device: torch.device) -> str:
    """로그와 리포트에 남길 장치 설명 문자열."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
