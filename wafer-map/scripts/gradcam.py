"""Grad-CAM: CNN이 맵의 어느 부분을 보고 판단했는지 계산한다 (PROTOCOL.md 8절).

마지막 합성곱 블록의 지도(8×8, 128장) 각각에 대해 "이 지도가 커지면 목표 클래스 점수가 얼마나 오르나"(기울기)를 구하고,
그 값을 가중치로 지도들을 더해 64×64로 키운다. 값이 클수록 그 위치가 판단에 크게 기여한 곳이다.
LightGBM과 같은 프로세스에서 돌지 않도록 따로 둔 스크립트다.

입력: maps.npy (개수, 2, 64, 64) uint8, targets.npy (개수,) 클래스 번호
출력: cams.npy (개수, 64, 64), 0~1로 정규화

실행: python scripts/gradcam.py <maps.npy> <targets.npy> <cams.npy>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cnn import WaferCNN, device  # noqa: E402


def grad_cam(model: WaferCNN, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    feats = model.features(x)              # (배치, 128, 8, 8)
    feats.retain_grad()
    scores = model.head(feats)             # (배치, 9)
    scores.gather(1, target[:, None]).sum().backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)          # 지도마다 기울기 평균
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))     # 목표 클래스 쪽으로 민 부분만
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).max(dim=1).values.clamp_min(1e-8)[:, None, None]
    return cam.detach().cpu().numpy()


def main() -> None:
    maps, targets, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dev = device()
    model = WaferCNN().to(dev)
    model.load_state_dict(torch.load(ROOT / "models" / "cnn_final.pt", map_location=dev))
    model.eval()
    X = torch.from_numpy(np.load(maps)).float().to(dev)
    t = torch.from_numpy(np.load(targets)).long().to(dev)
    cams = [grad_cam(model, X[i:i + 64], t[i:i + 64]) for i in range(0, len(X), 64)]
    np.save(out, np.concatenate(cams))


if __name__ == "__main__":
    main()
