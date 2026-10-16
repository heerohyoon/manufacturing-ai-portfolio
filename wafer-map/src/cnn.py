"""M2 CNN 정의와 학습 (PROTOCOL.md 3절). LightGBM과 같은 프로세스에서 쓰지 않는다 (src/models.py 참고)."""
import numpy as np
import torch
from torch import nn

from src.evaluation import N_CLASSES, macro_f1


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """[합성곱 3×3 → 배치 정규화 → ReLU] × 2 → 최대 풀링. 지도 크기는 절반, 지도 장수는 c_out."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WaferCNN(nn.Module):
    """입력 (배치, 2, 64, 64) → 64→32→16→8 크기, 지도 32→64→128장 → 평균 → 9개 점수."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(conv_block(2, 32), conv_block(32, 64), conv_block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def class_weights(y: np.ndarray) -> torch.Tensor:
    """(전체 수 / (클래스 수 × 그 클래스 수))의 제곱근. 드문 클래스를 키우되 54개짜리에 수백 배를 주지 않게."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = np.sqrt(len(y) / (N_CLASSES * np.maximum(counts, 1)))
    return torch.tensor(w, dtype=torch.float32)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """학습 때만: 맵마다 90도 회전 0~3회, 좌우 뒤집기 50%. 패턴 이름은 방향과 무관하다."""
    k = torch.randint(0, 4, (x.shape[0],), generator=gen)
    flip = torch.rand(x.shape[0], generator=gen) < 0.5
    out = torch.empty_like(x)
    for r in range(4):
        idx = (k == r).nonzero(as_tuple=True)[0].to(x.device)
        if len(idx):
            out[idx] = torch.rot90(x[idx], r, dims=(2, 3))
    fidx = flip.nonzero(as_tuple=True)[0].to(x.device)
    if len(fidx):
        out[fidx] = torch.flip(out[fidx], dims=(3,))
    return out


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(dev).float()
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


def train_cnn(X: np.ndarray, y: np.ndarray, epochs: int, X_val=None, y_val=None, patience: int = 5,
              seed: int = 0, batch: int = 256, log=print):
    """CNN을 학습한다. 검증 데이터가 있으면 에폭마다 매크로 F1을 재고, patience만큼 안 오르면 멈춘다.

    반환: (모델, 에폭별 기록, 가장 좋았던 에폭 수). 검증 데이터가 없으면 정해진 에폭 수만큼 학습한다.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = device()
    model = WaferCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y).to(dev))
    X_t, y_t = torch.from_numpy(np.array(X, copy=True)), torch.from_numpy(np.array(y, copy=True)).long()

    history, best_f1, best_epoch, best_state, waited = [], -1.0, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_t), generator=gen)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = augment(X_t[idx].to(dev).float(), gen)
            yb = y_t[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        record = {"epoch": epoch, "train_loss": total / len(X_t)}

        if X_val is not None:
            f1 = macro_f1(y_val, predict_proba(model, X_val).argmax(1))
            record["val_macro_f1"] = f1
            if f1 > best_f1:
                best_f1, best_epoch, waited = f1, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
        history.append(record)
        log(record)
        if X_val is not None and waited >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch
