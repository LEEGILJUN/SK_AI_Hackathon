"""`scripts/` 의 모든 스크립트가 임포트되는가.

**보내기 전에 여기서 걸려야 한다.** 다른 장비에 지시를 보냈는데 임포트에서
죽으면 그쪽은 5~13분을 버리고, 자기가 뭘 잘못했나 먼저 의심한다.

실제로 있었던 일이다.

    scripts/run_demo.py              저장소에 만들어진 적이 없는데 세 번 지시
    scripts/measure_baseline_map.py  `inspection.visa` 를 임포트 — 그런 모듈이 없다

둘째 것은 보내기 직전에 `scripts/check_handoff.py` 가 잡았다. 이 시험은 그
검사를 상시로 돌린다.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p for p in (REPO_ROOT / "scripts").glob("*.py")
                 if not p.name.startswith("_"))


def test_there_are_scripts_to_check():
    """스크립트를 하나도 못 찾으면 이 시험 자체가 무의미하다."""
    assert len(SCRIPTS) >= 10, f"스크립트를 {len(SCRIPTS)}개만 찾았다"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_a_script_imports(path):
    """임포트만 해 본다. **부수 효과 없이** 최상위가 도는지 보는 것이다.

    무거운 작업은 `main()` 안에 있어야 한다. 최상위에서 모델을 내려받거나
    공장을 세우면 `--help` 조차 몇 분씩 걸린다.
    """
    done = subprocess.run(
        [sys.executable, "-c",
         f"import importlib.util,sys; sys.path.insert(0,{str(REPO_ROOT)!r}); "
         f"spec=importlib.util.spec_from_file_location('probe',{str(path)!r}); "
         f"mod=importlib.util.module_from_spec(spec); "
         f"spec.loader.exec_module(mod)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, cwd=REPO_ROOT,
    )
    assert done.returncode == 0, (
        f"{path.name} 이 임포트되지 않는다:\n{done.stderr[-1200:]}"
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_a_script_does_not_run_on_import(path):
    """`if __name__ == "__main__":` 없이 최상위에서 실행되면 안 된다.

    임포트만 했는데 도는 스크립트는 시험에서도 돌고, 남의 장비에서도
    예상 못 한 때에 돈다.
    """
    source = path.read_text(encoding="utf-8")
    if "def main(" not in source:
        pytest.skip("main() 이 없는 스크립트")
    assert '__name__ == "__main__"' in source, (
        f"{path.name} 에 진입점 가드가 없다"
    )
