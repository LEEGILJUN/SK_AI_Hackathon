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


@pytest.fixture(scope="session")
def shaped_catalog():
    """실제 공장과 같은 배분의 레코드 목록. **이미지는 없다.**

    Mac 의 합성 공장은 로트가 14장이라 규모 때문에 갈리는 결함을 못 잡는다.
    자세한 것은 `tests/factory_shape.py` 를 보라.
    """
    from tests.factory_shape import build_catalog

    return build_catalog()


@pytest.fixture(scope="session")
def shaped_lookup(shaped_catalog):
    """실제 규모의 카탈로그를 보는 목 조회 계층."""
    from lookup import MockLookup
    from tests.factory_shape import LINES

    return MockLookup(
        catalog=shaped_catalog,
        banks={(line, obj): f"{obj}-{line[-2:]}-v1" for line, obj in LINES.items()},
    )
