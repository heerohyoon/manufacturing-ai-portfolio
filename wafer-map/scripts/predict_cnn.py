"""고정된 CNN 가중치로 예측 확률을 계산한다. LightGBM과 같은 프로세스에서 돌지 않도록 따로 둔 스크립트다.

입력: (개수, 2, 64, 64) uint8 배열을 담은 .npy 파일
출력: (개수, 9) 예측 확률 .npy 파일

실행: python scripts/predict_cnn.py <입력.npy> <출력.npy> [가중치.pt]
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cnn import WaferCNN, device, predict_proba  # noqa: E402


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    weights = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "models" / "cnn_final.pt"
    model = WaferCNN().to(device())
    model.load_state_dict(torch.load(weights, map_location=device()))
    np.save(dst, predict_proba(model, np.load(src)))


if __name__ == "__main__":
    main()
