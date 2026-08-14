"""학습 이력 인덱서 — 폴더만 보고 뱅크 구성을 복원한다.

`lookup/factory.py` 와 다른 문제를 푼다. 그쪽은 `manifest.csv` 를 읽는다 —
누군가 대장을 써 뒀다는 전제다. **여기는 대장이 없을 때** 파일만 보고
되짚는다. 현장의 기본값이 그쪽이다.

여기서 지키는 것 셋.

  1. 폴더 구조를 전제하지 않는다
  2. 복원한 것은 **추정으로 표시**한다. 확정 승격은 사람이 한다
  3. 왜 그렇게 봤는지 근거를 남긴다
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from indexer import scan_history
from indexer.scan import BANK_ARRAYS, BANK_META, summarise

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bank():
    """진짜 뱅크 하나. 합성이라 빠르다."""
    from app.pipeline import DemoFactory

    try:
        factory = DemoFactory(visa_root=Path(__file__).parent / "_no_visa_here")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return next(iter(factory.items.values())).bank


def test_the_file_names_match_what_the_bank_actually_writes():
    """인덱서가 찾는 이름과 뱅크가 쓰는 이름이 같다.

    두 벌이 되면 **인덱서가 뱅크를 하나도 못 찾는데 오류도 안 난다** —
    빈 결과가 정상처럼 보인다. 순환 import 를 피하려고 값을 맞춰 뒀으므로
    여기서 대조한다.
    """
    import inspection.bank as bank_module

    assert BANK_META == bank_module._BANK_META
    assert BANK_ARRAYS == bank_module._BANK_ARRAYS


def test_a_bank_with_its_record_is_not_an_estimate(tmp_path, bank):
    """이력 파일이 있으면 그대로 읽는다. 추정이 아니다."""
    bank.save(tmp_path / "아무" / "이름" / "이나")

    result = scan_history(tmp_path)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.confidence == "recorded"
    assert record.is_estimated is False
    assert record.images == bank.images
    assert record.patch_count == len(bank)
    assert any("추정이 아니다" in line for line in record.evidence)


def test_the_folder_layout_is_not_assumed(tmp_path, bank):
    """폴더 이름도 깊이도 가정하지 않는다.

    사내 폴더 구조를 전제하면 다른 현장에 못 쓴다.
    """
    bank.save(tmp_path / "a")
    deep = tmp_path / "전혀" / "다른" / "구조" / "여덟" / "단계" / "아래"
    other = copy.deepcopy(bank)
    other.meta = dict(other.meta, bank_version="다른-버전")
    other.save(deep)

    result = scan_history(tmp_path)

    assert {r.bank_version for r in result.records} == {
        bank.meta["bank_version"], "다른-버전"
    }


def test_a_bank_without_its_record_is_marked_as_an_estimate(tmp_path, bank):
    """벡터만 있고 이력이 없으면 추정이다. 이웃 이미지는 **후보**다.

    여기서 확정으로 올리면 담당자가 확인할 방법이 없다. 역추정한 이력을
    확정처럼 쓰는 것이 이 과제가 막으려는 것 중 하나다.
    """
    bank.save(tmp_path / "정상")
    lost = tmp_path / "이력없음"
    lost.mkdir()
    shutil.copy(tmp_path / "정상" / BANK_ARRAYS, lost / BANK_ARRAYS)
    (tmp_path / "img_001.png").write_bytes(b"")

    result = scan_history(tmp_path)
    inferred = [r for r in result.records if r.is_estimated]

    assert len(inferred) == 1
    record = inferred[0]
    assert record.confidence == "inferred"
    assert record.images, "후보를 하나도 못 찾았다"
    assert any("후보일 뿐" in line for line in record.evidence)
    assert record.to_profile().is_estimated is True


def test_the_version_diff_names_what_was_removed(tmp_path, bank):
    """재구성 전후에 무엇이 빠졌는지를 파일명까지 짚는다.

    **승인의 근거가 이것이다.** "정상 이미지 몇 장을 함께 버렸는가"에
    답할 수 있어야 사람이 판단한다. 숫자만으로는 판단할 수 없다.
    """
    bank.save(tmp_path / "v1")
    after = copy.deepcopy(bank)
    after.meta = dict(after.meta, bank_version="그다음")
    dropped = bank.images[-3:]
    after.images = bank.images[:-3]
    after.save(tmp_path / "v2")

    result = scan_history(tmp_path)
    diff = result.diff(bank.meta["bank_version"], "그다음")

    assert diff is not None
    assert sorted(diff.removed) == sorted(dropped)
    assert diff.added == []
    assert diff.kept == len(bank.images) - 3
    assert "3장 제거" in diff.describe()


def test_a_diff_touching_an_estimate_is_itself_an_estimate(tmp_path, bank):
    """한쪽이 추정이면 비교도 추정이다. 그 표시가 사라지면 안 된다."""
    bank.save(tmp_path / "v1")
    lost = tmp_path / "이력없음"
    lost.mkdir()
    shutil.copy(tmp_path / "v1" / BANK_ARRAYS, lost / BANK_ARRAYS)

    result = scan_history(tmp_path)
    diff = result.diff(bank.meta["bank_version"], "이력없음")

    assert diff is not None
    assert diff.is_estimated is True
    assert "담당자 확인" in diff.describe()


def test_a_recorded_bank_wins_over_an_estimated_one(tmp_path, bank):
    """같은 버전이 두 자리에 있으면 확정을 남기고 근거에 적는다.

    조용히 하나를 고르면 왜 그쪽을 골랐는지 알 수 없다.
    """
    bank.save(tmp_path / "제대로")
    ghost = tmp_path / bank.meta["bank_version"]
    ghost.mkdir()
    shutil.copy(tmp_path / "제대로" / BANK_ARRAYS, ghost / BANK_ARRAYS)

    result = scan_history(tmp_path)
    record = result.by_version(bank.meta["bank_version"])

    assert record is not None
    assert record.confidence == "recorded"
    assert any("버렸다" in line for line in record.evidence)
    assert result.skipped


def test_a_capped_coreset_is_surfaced(tmp_path, bank):
    """coreset 이 상한에 걸린 것을 근거에 남긴다.

    조용히 잘리면 "비율대로 만들어졌다"고 오해한다. 실제로
    `coreset_ratio 0.1` 요청이 상한에 걸려 4.1% 였던 적이 있다.
    """
    capped = copy.deepcopy(bank)
    capped.meta = dict(
        capped.meta, coreset_capped=True, coreset_ratio=0.1, max_bank_size=20000
    )
    capped.save(tmp_path / "capped")

    record = scan_history(tmp_path).records[0]
    assert any("상한에 걸렸다" in line for line in record.evidence)


def test_nothing_found_is_not_an_error(tmp_path):
    """없는 폴더나 빈 폴더에서 예외를 내지 않는다."""
    assert scan_history(tmp_path / "없는곳").records == []
    assert scan_history(tmp_path).records == []
    assert "찾지 못했다" in summarise(scan_history(tmp_path))


def test_the_summary_says_which_are_estimates(tmp_path, bank):
    """사람이 읽는 요약에 확정·추정이 구분돼 나온다."""
    bank.save(tmp_path / "v1")
    lost = tmp_path / "이력없음"
    lost.mkdir()
    shutil.copy(tmp_path / "v1" / BANK_ARRAYS, lost / BANK_ARRAYS)

    text = summarise(scan_history(tmp_path))
    assert "[확정]" in text and "[추정]" in text
    assert "담당자 확인 후에만" in text
