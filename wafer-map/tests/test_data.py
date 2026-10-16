"""평가 데이터 잠금 규칙을 확인한다.

실행 (wafer-map/ 폴더에서): python -m pytest -q
"""
from pathlib import Path

import pytest

from src import data

ROOT = Path(__file__).resolve().parents[1]
FINAL_SCRIPT = ROOT / "scripts" / "final_evaluation.py"
# 최종 평가가 끝난 뒤 추가한 사후 분석 스크립트 (PROTOCOL.md 변경 기록). 결정을 바꾸지 않는다.
POSTHOC_SCRIPT = ROOT / "scripts" / "posthoc_predictions.py"


def test_load_test_requires_unlock():
    with pytest.raises(PermissionError):
        data.load_test(unlock="")


def test_only_final_script_opens_test():
    """load_test를 부르는 파일은 최종 평가 스크립트 하나뿐이어야 한다 (노트북 포함)."""
    callers = []
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.ipynb")):
        if ".venv" in path.parts or path.name in {"data.py", "test_data.py"}:
            continue
        if "load_test(" in path.read_text(encoding="utf-8"):
            callers.append(path.relative_to(ROOT))
    assert set(callers) <= {FINAL_SCRIPT.relative_to(ROOT), POSTHOC_SCRIPT.relative_to(ROOT)}, callers


def test_no_other_file_reads_locked_pickle():
    """잠금 함수를 거치지 않고 평가 파일을 직접 읽는 코드도 막는다."""
    allowed = {"data.py", "test_data.py", "prepare_data.py"}
    offenders = [p.relative_to(ROOT) for p in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.ipynb"))
                 if ".venv" not in p.parts and p.name not in allowed
                 and "test_locked" in p.read_text(encoding="utf-8")]
    assert not offenders, offenders
