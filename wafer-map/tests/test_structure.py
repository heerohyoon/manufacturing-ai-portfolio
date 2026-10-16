"""LightGBM 쪽 코드와 PyTorch 쪽 코드가 한 프로그램에서 섞이지 않는지 확인한다.

macOS에서 둘을 한 프로세스에서 쓰면 LightGBM 학습이 멈추거나 죽는다 (src/models.py 참고).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LGBM = re.compile(r"^\s*(from|import)\s+(lightgbm|src\.models)\b", re.M)
TORCH = re.compile(r"^\s*(from|import)\s+(torch|src\.cnn)\b", re.M)


def test_no_file_imports_both():
    both = [p.relative_to(ROOT) for p in list((ROOT / "src").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
            if LGBM.search(t := p.read_text(encoding="utf-8")) and TORCH.search(t)]
    assert not both, both


def test_shared_modules_import_neither():
    for name in ["evaluation.py", "data.py", "preprocess.py", "features.py"]:
        text = (ROOT / "src" / name).read_text(encoding="utf-8")
        assert not LGBM.search(text) and not TORCH.search(text), name
