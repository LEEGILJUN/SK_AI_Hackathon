"""시험 전체가 함께 쓰는 준비물.

**`DemoFactory()` 는 뱅크를 세운다.** 합성 데이터에서는 1초도 안 걸리지만
VisA 가 있는 장비에서는 한 번에 110초가 넘는다. 시험마다 새로 만들면 그 시험
하나가 2분씩 먹는다 — 4090 실측에서 여섯 시험이 731초로 전체의 95% 였다.

한 번 세워 돌려 쓴다. 시험이 공장을 고치지 않으므로 나눠 써도 안전하다.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def demo_factory():
    """시연용 가상 공장. 시험 전체에서 한 번만 세운다."""
    from app.pipeline import DemoFactory

    return DemoFactory()


@pytest.fixture(scope="session")
def demo_lookup(demo_factory):
    """`demo_factory` 를 보는 목 조회 계층."""
    from lookup import MockLookup

    return MockLookup(
        catalog=demo_factory.catalog,
        banks=demo_factory.bank_versions(),
        quality_provider=demo_factory.quality_baseline,
    )
