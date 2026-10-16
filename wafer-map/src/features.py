"""M0 크기 특징과 M1 모양 특징 (PROTOCOL.md 3절)."""
import numpy as np
import pandas as pd
from scipy import ndimage

from src.preprocess import SIZE

# 64×64 격자에서 중심 거리(0 = 중심, 1 = 가장자리)와 방향(라디안)을 미리 계산해 둔다
_yy, _xx = np.mgrid[0:SIZE, 0:SIZE]
_CENTER = (SIZE - 1) / 2
RADIUS = np.sqrt(((_yy - _CENTER) / (SIZE / 2)) ** 2 + ((_xx - _CENTER) / (SIZE / 2)) ** 2)
ANGLE = np.arctan2(_yy - _CENTER, _xx - _CENTER)

RING_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
N_SECTORS = 8
OUTER_RADIUS = 0.6


def size_features(maps) -> pd.DataFrame:
    """M0: 원래 맵의 세로, 가로, 칩 개수. 불량 정보는 쓰지 않는다. 사실상 제품 종류를 나타낸다."""
    return pd.DataFrame({
        "height": [m.shape[0] for m in maps],
        "width": [m.shape[1] for m in maps],
        "n_dies": [int((m > 0).sum()) for m in maps],
    })


def _rate(fail: np.ndarray, die: np.ndarray, region: np.ndarray) -> float:
    n = (die & region).sum()
    return float((fail & region).sum() / n) if n else 0.0


def shape_features_one(channels: np.ndarray) -> dict:
    """64×64 맵 하나의 모양 특징. 원래 크기와 칩 개수는 넣지 않는다 (크기 지름길을 막기 위해)."""
    die, fail = channels[0].astype(bool), channels[1].astype(bool)
    n_die, n_fail = die.sum(), fail.sum()
    out = {"fail_rate": float(n_fail / n_die) if n_die else 0.0}

    # 중심에서의 거리로 나눈 고리별 불량 비율: Center, Donut, Edge-Ring을 가른다
    for i, (a, b) in enumerate(zip(RING_EDGES[:-1], RING_EDGES[1:])):
        out[f"ring_{i}"] = _rate(fail, die, (RADIUS >= a) & (RADIUS < b))

    # 바깥쪽을 부채꼴 8개로 나눈 불량 비율: 가장자리 일부(Edge-Loc)인지 전체(Edge-Ring)인지 가른다
    sector = ((ANGLE + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    rates = np.array([_rate(fail, die, (RADIUS >= OUTER_RADIUS) & (sector == s)) for s in range(N_SECTORS)])
    for s, r in enumerate(np.sort(rates)[::-1]):  # 방향 자체는 의미가 없으므로 큰 순서로 정렬해 회전에 무관하게
        out[f"sector_rank_{s}"] = float(r)
    out["sector_std"] = float(rates.std())
    out["sector_range"] = float(rates.max() - rates.min())

    # 이어진 불량 덩어리: Loc(한 덩어리), Scratch(길쭉함), Random(작은 덩어리 여럿)을 가른다
    # 대각선 긁힘은 모서리끼리만 닿으므로 대각선 방향까지 이어진 것으로 본다 (8방향)
    labels, n_comp = ndimage.label(fail, structure=np.ones((3, 3), dtype=int))
    out["n_clusters_per_die"] = float(n_comp / n_die) if n_die else 0.0
    if n_comp:
        sizes = np.bincount(labels.ravel())[1:]
        big = labels == (np.argmax(sizes) + 1)
        ys, xs = np.nonzero(big)
        out["largest_share"] = float(sizes.max() / n_fail)
        out["largest_radius"] = float(RADIUS[ys, xs].mean())
        if len(ys) >= 3:
            eig = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs])))
            out["largest_elongation"] = float(np.sqrt(eig[1] / max(eig[0], 1e-3)))
        else:
            out["largest_elongation"] = 1.0
    else:
        out.update({"largest_share": 0.0, "largest_radius": 0.0, "largest_elongation": 1.0})
    return out


def shape_features(channels: np.ndarray) -> pd.DataFrame:
    """(개수, 2, 64, 64) 배열 → 맵마다 모양 특징 한 줄."""
    return pd.DataFrame([shape_features_one(c) for c in channels])
