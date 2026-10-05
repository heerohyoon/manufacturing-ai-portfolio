# Manufacturing AI Portfolio

반도체 제조 데이터에 AI를 적용하는 두 가지 대표 문제를 다룹니다.

| 프로젝트 | 데이터 | 문제 | 핵심 기술 |
|---|---|---|---|
| [`secom/`](secom/) | UCI SECOM (공정 센서 590개) | 공정 센서 기반 불량 예측 | 불균형 분류, 결측 처리, LightGBM, SHAP |
| [`wafer-map/`](wafer-map/) | WM-811K (예정) | 웨이퍼 맵 불량 패턴 분류 | CNN (예정) |

## 환경 세팅

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r secom/requirements.txt
# macOS: LightGBM은 OpenMP가 필요합니다 → brew install libomp
```

정확히 재현하려면 `secom/requirements.lock.txt`(pip freeze 결과)를 사용하세요.
