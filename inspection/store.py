"""라인·품목별 뱅크 저장소 — 보관하고, 비교하고, 되돌린다.

지금까지 뱅크는 **어디에도 저장되지 않았다.** `DemoFactory` 가 프로세스마다
메모리에 새로 세웠고(4090 실측 108초), 재구성한 뱅크는 배포 패키지 안에만
들어갔다. 그래서 **새 뱅크의 성능이 나쁠 때 돌아갈 파일이 없었다.**

    banks/
      pcb1-01/                          라인·품목 (lookup.base.bank_item_key)
        v1_20260815-1430_a3f2c1/        버전 · 만든 시각 · 설정 지문
          bank.npz
          bank_meta.json
        v2_20260816-0910_a3f2c1/
        CURRENT                         "v1_20260815-1430_a3f2c1" 한 줄

── 이름에 세 조각을 넣는 이유 ──────────────────────────────────────────

**버전** — 계보다. 재구성 기록의 `rebuilt_from` 이 이 값을 가리키므로,
되돌릴 대상이 하나로 정해진다.

**만든 시각** — 사람이 읽고 정렬한다. 같은 `v2` 가 맥과 4090 에서 따로
나올 수 있는데, 시각이 없으면 어느 것이 무엇인지 폴더를 열어봐야 안다.

**설정 지문 6자리** — **이름만 보고 못 쓰는 뱅크를 거른다.** 2026-08-15 에
입력을 448 에서 512 로 바꿨는데, 그 전 뱅크는 거리 척도가 달라 쓸 수 없다.
지문이 이름에 없으면 폴더를 열어 메타를 읽어야 알 수 있고, 실제로 옛 뱅크가
검사를 통과해 조용히 틀린 점수를 낸 적이 있다.

**설정 전체를 이름에 넣지 않는다.** 이름이 길어지고, 설정 항목이 하나 늘 때마다
규칙이 또 바뀐다. **판정은 이름이 아니라 `bank_meta.json` 의 지문으로 한다**
(`MemoryBank.assert_compatible`). 이름의 지문은 사람이 훑을 때와 후보를 빨리
줄일 때 쓰는 것이지 검증 수단이 아니다.

── CURRENT 를 바꾸는 것은 사람이 한다 ──────────────────────────────────

저장은 자동이고 **전환은 아니다.** `CURRENT` 가 가리키는 것이 실제 판정에
쓰이는 뱅크이므로, 그것을 코드가 바꾸면 무인 배포가 된다. `write_current` 는
사람이 실행하는 스크립트에서만 부른다 — `tests/test_bank_store.py` 가
`agents/` 와 `scheduler/` 에서 이 함수를 부르지 않는지 검사한다.

되돌리기도 같은 함수다. **파일을 옮기지 않고 가리키는 이름만 되돌린다.**
그래서 원복이 실패할 여지가 없다.

심볼릭 링크를 쓰지 않는 이유는 시연 장비가 Windows 라서다. 링크 생성에
권한이 필요하고, 없으면 조용히 복사본이 생긴다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .bank import MemoryBank
from .features import FeatureConfig

#: 저장소 뿌리. 저장소 자체는 `.gitignore` 에 있다 — 뱅크 파일은 커밋하지 않는다.
DEFAULT_STORE_ROOT = Path("banks")

#: 어느 판이 운영에 쓰이는가. 한 줄짜리 텍스트다.
CURRENT_FILE = "CURRENT"

_FOLDER = re.compile(r"^(v\d+)_(\d{8}-\d{4})_([0-9a-f]{6})$")
_TIME_FORMAT = "%Y%m%d-%H%M"


def config_id(config: FeatureConfig) -> str:
    """설정 지문을 6자리로 줄인다. 이름에 넣을 용도다.

    **충돌을 걱정할 값이 아니다.** 설정 조합이 수십 개를 넘지 않고, 실제
    검증은 `assert_compatible` 이 지문 전체를 대조해서 한다. 여기서 6자리는
    "이 폴더는 지금 설정과 다르다"를 폴더를 열지 않고 알아채기 위한 것이다.
    """
    payload = json.dumps(config.fingerprint(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:6]


def folder_name(version: str, built_at: datetime, config: FeatureConfig) -> str:
    """저장 폴더 이름. 세 조각을 밑줄로 잇는다."""
    return f"{version}_{built_at.strftime(_TIME_FORMAT)}_{config_id(config)}"


@dataclass(frozen=True)
class StoredBank:
    """저장소에 있는 뱅크 한 판. **벡터를 읽지 않고 이름만으로 만든다.**

    목록을 훑을 때 뱅크 파일을 전부 여는 것은 느리고, 대개는 이름에 있는
    것만으로 충분하다. 실제로 쓸 때 `load()` 한다.
    """

    path: Path
    item_key: str
    version: str
    built_at: datetime
    config_id: str
    is_current: bool = False

    @property
    def version_number(self) -> int:
        return int(self.version[1:])

    def matches(self, config: FeatureConfig) -> bool:
        """지금 설정으로 쓸 수 있는 판인가 — 이름만 보고 판단한다."""
        return self.config_id == config_id(config)

    def load(self, config: FeatureConfig | None = None) -> MemoryBank:
        """벡터까지 읽는다. 설정을 주면 지문 전체를 대조한다."""
        bank = MemoryBank.load(self.path)
        if config is not None:
            bank.assert_compatible(config)
        return bank


def _item_dir(root: str | Path, item_key: str) -> Path:
    return Path(root) / item_key


def save_bank(
    bank: MemoryBank,
    item_key: str,
    root: str | Path = DEFAULT_STORE_ROOT,
    config: FeatureConfig | None = None,
    built_at: datetime | None = None,
) -> Path:
    """뱅크 한 판을 저장한다. **CURRENT 는 건드리지 않는다.**

    저장과 전환은 다른 일이다. 새로 만든 뱅크가 곧바로 판정에 쓰이면
    게이트도 섀도도 지나지 않은 것이 운영에 들어간다.

    설정을 안 주면 뱅크 메타에 적힌 것을 쓴다. 뱅크가 자기 설정을 들고
    있으므로 부르는 쪽이 다시 알려줄 필요가 없다.
    """
    if config is None:
        stored = bank.meta.get("feature_config") or {}
        layers = stored.get("layers")
        config = FeatureConfig(
            backbone=stored.get("backbone", FeatureConfig.backbone),
            layers=tuple(layers) if layers else FeatureConfig.layers,
            weights=stored.get("weights", FeatureConfig.weights),
            crop=int(stored.get("crop", FeatureConfig.crop)),
            neighborhood=int(stored.get("neighborhood", FeatureConfig.neighborhood)),
        )

    built_at = built_at or datetime.now()
    directory = _item_dir(root, item_key) / folder_name(bank.version, built_at, config)
    bank.save(directory)
    return directory


def list_banks(
    item_key: str,
    root: str | Path = DEFAULT_STORE_ROOT,
    config: FeatureConfig | None = None,
) -> list[StoredBank]:
    """이 품목의 저장된 판 목록. **최신 버전이 앞이다.**

    설정을 주면 그 설정으로 쓸 수 있는 것만 남긴다. 이름 규칙에 맞지 않는
    폴더는 조용히 건너뛴다 — 사람이 손으로 만든 폴더가 섞여도 무너지지 않아야
    한다.
    """
    directory = _item_dir(root, item_key)
    if not directory.is_dir():
        return []

    current = _read_pointer(directory)
    found: list[StoredBank] = []
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        match = _FOLDER.match(child.name)
        if match is None:
            continue
        version, stamp, fingerprint = match.groups()
        found.append(
            StoredBank(
                path=child,
                item_key=item_key,
                version=version,
                built_at=datetime.strptime(stamp, _TIME_FORMAT),
                config_id=fingerprint,
                is_current=(child.name == current),
            )
        )

    if config is not None:
        found = [b for b in found if b.matches(config)]

    found.sort(key=lambda b: (b.version_number, b.built_at), reverse=True)
    return found


def _read_pointer(directory: Path) -> str | None:
    pointer = directory / CURRENT_FILE
    if not pointer.is_file():
        return None
    return pointer.read_text(encoding="utf-8").strip() or None


def current_bank(
    item_key: str,
    root: str | Path = DEFAULT_STORE_ROOT,
    config: FeatureConfig | None = None,
) -> StoredBank | None:
    """지금 판정에 쓰는 판. 없으면 None.

    설정을 주면 **지문이 안 맞을 때도 None 이 아니라 그 판을 돌려준다.**
    부르는 쪽이 "가리키는 것은 있는데 못 쓴다"와 "가리키는 것이 없다"를
    구분해야 하기 때문이다. 앞은 재구성이 필요한 상황이고 뒤는 최초 구성이다.
    """
    directory = _item_dir(root, item_key)
    name = _read_pointer(directory)
    if name is None:
        return None
    for stored in list_banks(item_key, root):
        if stored.path.name == name:
            return stored
    return None


def write_current(item_key: str, folder: str, root: str | Path = DEFAULT_STORE_ROOT) -> Path:
    """운영에 쓸 판을 지정한다. **사람이 실행하는 자리다.**

    에이전트 코드에서 부르지 않는다. 이 함수가 가리키는 것을 바꾸는 순간
    실제 판정에 쓰이는 뱅크가 바뀌고, 그것은 배포다. 릴리즈 에이전트가
    승인 요청 문서까지만 만드는 것과 같은 경계다.

    되돌리기도 이 함수다. 이전 폴더 이름을 다시 써 넣으면 끝이고, 파일은
    움직이지 않는다.
    """
    directory = _item_dir(root, item_key)
    if not (directory / folder).is_dir():
        available = ", ".join(b.path.name for b in list_banks(item_key, root)) or "없음"
        raise FileNotFoundError(
            f"{item_key} 에 '{folder}' 판이 없다. 있는 것: {available}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / CURRENT_FILE
    pointer.write_text(folder + "\n", encoding="utf-8")
    return pointer


def load_current(
    item_key: str,
    root: str | Path = DEFAULT_STORE_ROOT,
    config: FeatureConfig | None = None,
) -> MemoryBank | None:
    """운영 뱅크를 읽는다. 없거나 설정이 안 맞으면 None.

    **여기서는 예외를 던지지 않는다.** 저장소에 쓸 만한 것이 없다는 것은
    오류가 아니라 "새로 세워야 한다"는 뜻이고, 부르는 쪽이 그렇게 하면 된다.
    """
    stored = current_bank(item_key, root)
    if stored is None:
        return None
    if config is not None and not stored.matches(config):
        return None
    try:
        return stored.load(config)
    except (FileNotFoundError, ValueError):
        return None
