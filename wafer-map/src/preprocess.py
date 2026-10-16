"""웨이퍼 맵 크기 맞추기 (PROTOCOL.md 2절)."""
import numpy as np

SIZE = 64


def _block_starts(n: int, size: int) -> np.ndarray:
    """출력 칸 i가 원래 축에서 시작하는 위치. 원래 축이 더 짧으면 같은 칸이 반복된다."""
    return (np.arange(size) * n) // size


def resize_blockmax(m: np.ndarray, size: int = SIZE) -> np.ndarray:
    """각 출력 칸에 대응하는 원래 칸들 중 가장 큰 값을 쓴다 (0 칩 없음 < 1 정상 < 2 불량).

    원래 축이 size보다 짧으면 칸을 그대로 늘린 것과 같고,
    길면 합쳐지는 칸 중 불량이 하나라도 있으면 불량이 된다. 큰 맵의 가는 선(Scratch)을 지키기 위해서다.
    np.maximum.reduceat은 시작 위치가 같거나 줄어드는 곳에서는 그 칸 하나만 쓰므로 늘리는 경우도 그대로 처리된다.
    """
    rows = np.maximum.reduceat(m, _block_starts(m.shape[0], size), axis=0)
    return np.maximum.reduceat(rows, _block_starts(m.shape[1], size), axis=1)


def to_channels(m64: np.ndarray) -> np.ndarray:
    """64×64 맵을 두 장의 0/1 지도로 바꾼다: [칩 위치, 불량 칩 위치]."""
    return np.stack([(m64 > 0), (m64 == 2)]).astype(np.uint8)


def stack_maps(maps) -> np.ndarray:
    """여러 맵을 (개수, 2, 64, 64) uint8 배열로."""
    return np.stack([to_channels(resize_blockmax(m)) for m in maps])
