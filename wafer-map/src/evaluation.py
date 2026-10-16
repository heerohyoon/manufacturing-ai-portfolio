"""교차검증 분할과 지표. LightGBM, PyTorch 어느 쪽도 불러오지 않는다."""
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data import CLASSES

N_CLASSES = len(CLASSES)
CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)  # 같은 로트는 같은 fold로


def macro_f1(y_true, y_pred) -> float:
    """9개 클래스 F1의 단순 평균. 한 번도 예측하지 않은 클래스는 F1 0으로 센다."""
    return float(f1_score(y_true, y_pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0))
