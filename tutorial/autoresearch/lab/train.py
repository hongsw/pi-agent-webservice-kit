#!/usr/bin/env python3
"""train.py — 에이전트가 수정하는 유일한 파일. karpathy/autoresearch의 train.py에 대응.

실물에서는 GPT 모델 + 옵티마이저 + 학습루프 전체가 여기 들어가고 "Everything is fair game".
이 mock에서는 그 전체를 MODEL knob과 짧은 가짜 학습으로 축약한다. 실행하면 학습된
모델을 JSON으로 stdout에 출력한다 → run_lab.py가 prepare.evaluate로 채점한다.

run_lab.py의 mutator(=Pi 에이전트 대역)가 아래 EDITABLE REGION만 고쳐 실험을 반복한다.
"""

from __future__ import annotations

import json

# ===== EDITABLE REGION (에이전트가 이 한 줄만 고친다) =====
MODEL = {'width': 256, 'depth': 4, 'dropout': 0.3, 'lr': 0.003}
# ===== END EDITABLE REGION =====


def build_and_train(steps: int = 1000) -> dict:
    """모델을 만들고 고정 예산(steps)으로 학습. mock: knob을 그대로 '학습된 모델'로 본다.

    실물에서는 여기서 모델/옵티마이저를 만들고 steps만큼 실제 학습한다. 학습 budget은
    karpathy 원본의 '고정 5분'에 대응(여기선 호출자가 steps로 고정).
    """
    # (mock) 학습은 knob을 안정화하는 정도의 의미만 — 평가는 prepare.py가 한다.
    trained = dict(MODEL)
    trained["_trained_steps"] = steps
    return trained


if __name__ == "__main__":
    model = build_and_train()
    # _trained_steps는 평가에 영향 주지 않도록 분리해 출력
    out = {k: v for k, v in model.items() if not k.startswith("_")}
    print(json.dumps(out))
