"""실물 backend — growing-memory 설계 4축 + SSL 표현축의 PyTorch 참조 구현.

업스트림 `growing-memory-pytorch`가 없을 때(GROWING_MEMORY_HOME 미설정), 이 패키지가
설계 축을 실제 torch 모듈로 실현해 **실제 MQAR 연관회상 과제로 학습/평가**한다.
업스트림이 있으면 model_adapter가 그쪽을 우선한다.

torch 필요(4090). torch 없으면 import 단계에서 실패 → adapter가 mock으로 폴백.
"""

from .model import GrowingMemoryModel, build_real
from .trainer import train_and_eval, RealRun

__all__ = ["GrowingMemoryModel", "build_real", "train_and_eval", "RealRun"]
