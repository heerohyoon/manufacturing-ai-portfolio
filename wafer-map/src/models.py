"""M0, M1 (LightGBM) 정의 (PROTOCOL.md 3절).

이 파일과 src/cnn.py는 한 프로그램 안에서 함께 쓰지 않는다. macOS에서 LightGBM과 PyTorch가 서로 다른
OpenMP 라이브러리를 쓰는데, 같은 프로세스에서 PyTorch가 CPU 연산을 한 뒤 LightGBM을 학습하면 멈추거나 죽는다.
그래서 CNN은 별도 스크립트(별도 프로세스)로 돌린다.
"""
from lightgbm import LGBMClassifier

from src.evaluation import CV, N_CLASSES, macro_f1  # noqa: F401  (다른 스크립트에서 같이 쓰기 위해 다시 내보냄)

M1_GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (15, 31) for mc in (20, 50)]


def build_m0() -> LGBMClassifier:
    """M0: 크기만 보는 기준선. 튜닝하지 않는다."""
    return LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=0, verbose=-1)


def build_m1(num_leaves: int = 31, min_child_samples: int = 20) -> LGBMClassifier:
    """M1: 모양 특징 + LightGBM."""
    return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=num_leaves,
                          min_child_samples=min_child_samples, class_weight="balanced",
                          random_state=0, verbose=-1)
