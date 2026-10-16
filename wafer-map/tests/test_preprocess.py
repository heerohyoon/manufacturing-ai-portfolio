"""64×64 변환과 모양 특징이 의도대로 동작하는지 확인한다."""
import numpy as np

from src.features import shape_features_one
from src.preprocess import SIZE, resize_blockmax, to_channels


def disc(n: int) -> np.ndarray:
    """n×n 원형 웨이퍼: 바깥 0, 안쪽 1."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    return (((yy - c) ** 2 + (xx - c) ** 2) <= (n / 2) ** 2).astype(np.int64)


def test_small_map_is_upsampled_without_change_in_values():
    m = disc(25)
    m[12, 12] = 2
    out = resize_blockmax(m)
    assert out.shape == (SIZE, SIZE)
    assert set(np.unique(out)) == {0, 1, 2}
    assert out[SIZE // 2, SIZE // 2] == 2


def test_thin_line_survives_downsampling():
    m = disc(200)
    m[100, 30:170] = 2          # 한 줄짜리 가로 선
    out = resize_blockmax(m)
    assert (out == 2).sum(axis=1).max() >= 40   # 줄인 뒤에도 한 행에 이어진 선으로 남아 있다


def test_non_square_map():
    m = np.ones((84, 212), dtype=np.int64)
    m[:, 100] = 2               # 세로 선 하나
    out = resize_blockmax(m)
    assert out.shape == (SIZE, SIZE)
    assert (out == 2).any(axis=0).sum() >= 1 and (out == 2).all(axis=0).sum() >= 1


def test_center_pattern_features():
    m = disc(64)
    yy, xx = np.mgrid[0:64, 0:64]
    m[((yy - 31.5) ** 2 + (xx - 31.5) ** 2) <= 36] = 2   # 가운데 덩어리
    f = shape_features_one(to_channels(resize_blockmax(m)))
    assert f["ring_0"] > 0.9 and f["ring_4"] == 0.0
    assert f["largest_share"] == 1.0 and f["largest_radius"] < 0.2


def test_scratch_is_elongated_and_rotation_invariant_sectors():
    m = disc(64)
    for i in range(10, 54):
        m[i, i] = 2             # 대각선 긁힘
    c = to_channels(resize_blockmax(m))
    f = shape_features_one(c)
    f_rot = shape_features_one(np.rot90(c, k=1, axes=(1, 2)).copy())
    assert f["largest_elongation"] > 5
    assert np.isclose(f["sector_std"], f_rot["sector_std"], atol=0.02)
