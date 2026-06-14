"""모델 어댑터 — build(cfg) / 동치테스트를 실물 growing-memory-pytorch에 위임하고,
없으면 순수 stdlib mock으로 폴백한다(설계 T0: torch/GPU 없이 루프 폐쇄 검증).

실물 통합 지점(seam):
  - REAL: `import growing_memory` 가 되면 그쪽 build/equivalence API를 호출.
  - MOCK: 패키지가 없으면 config 의존적인 "능력 신호"를 합성해 ASHA가 의미 있게
          상위 config를 골라낼 수 있도록 한다.

실물 패키지를 붙이려면 환경변수 GROWING_MEMORY_HOME 에 repo 경로를 주거나
`pip install growing-memory-pytorch` 후 _try_import_real()의 매핑만 맞추면 된다.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any


# ── 실물 런타임 설정(우리 torch 참조 구현용) ─────────────────────────────────
# loop가 스윕 시작 시 configure_real()로 채운다(과제/디바이스/데이터 규모).
REAL_RT: dict[str, Any] = {
    "device": "cuda",
    "task": "factory_mqar",
    "vocab": 64,
    "seq_len": 128,
    "num_pairs": 8,
    "batch": 32,
    "lr": 3e-3,
    "full_seq_len": 192,
    "seed": 0,
}


def configure_real(**kw) -> None:
    REAL_RT.update({k: v for k, v in kw.items() if v is not None})


def _real_backend_available() -> bool:
    """우리 torch 참조 구현 사용 가능?(torch 설치 + 강제 mock 아님)."""
    if os.environ.get("AR_FORCE_MOCK") == "1":
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ── 업스트림 패키지 탐지(seam) ───────────────────────────────────────────────
REAL_AVAILABLE = False
_real_mod = None


def _try_import_real():
    """growing-memory-pytorch를 import 시도. 성공 시 모듈 핸들 반환, 실패 시 None."""
    global REAL_AVAILABLE, _real_mod
    if _real_mod is not None:
        return _real_mod
    home = os.environ.get("GROWING_MEMORY_HOME")
    if home and os.path.isdir(home) and home not in sys.path:
        sys.path.insert(0, home)
    for name in ("growing_memory", "growing_memory_pytorch"):
        try:
            mod = __import__(name)
            _real_mod = mod
            REAL_AVAILABLE = True
            return mod
        except Exception:
            continue
    return None


@dataclass
class EqResult:
    ran: bool
    ok: bool
    reason: str = ""


def try_equivalence_test(cfg: dict[str, Any]) -> EqResult:
    """§3.4 동치/shape 테스트를 *작은 차원*으로 실행(실물만). 없으면 ran=False."""
    mod = _try_import_real()
    if mod is None:
        return EqResult(ran=False, ok=True, reason="no real package")
    # 실물 패키지가 동치 테스트 API를 노출한다고 가정한 호출 지점.
    # 패키지마다 이름이 다를 수 있어 후보를 순서대로 시도한다.
    small = dict(cfg)
    small["d_model"] = min(int(cfg.get("d_model", 64)), 64)
    small["n_layers"] = min(int(cfg.get("n_layers", 2)), 2)
    for fn_name in ("run_equivalence_test", "equivalence_test", "check_config"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                ok = bool(fn(small))
                return EqResult(ran=True, ok=ok, reason=fn_name)
            except Exception as e:  # 동치 깨짐/shape 오류 → 게이트 탈락
                return EqResult(ran=True, ok=False, reason=f"{fn_name}: {e}")
    # 패키지는 있지만 동치 API가 없으면 검증 불가 → skip 처리(구조 게이트로만)
    return EqResult(ran=False, ok=True, reason="real package lacks equivalence API")


# ── 모델 핸들 ────────────────────────────────────────────────────────────────
class ModelHandle:
    """학습/평가에 쓰이는 모델 핸들. backend는 'real' 또는 'mock'."""

    def __init__(self, cfg: dict[str, Any], backend: str, real_model: Any = None,
                 real_run: Any = None):
        self.cfg = cfg
        self.backend = backend
        self.real_model = real_model
        self.real_run = real_run        # 우리 torch 참조 구현(RealRun)
        self.trained_steps = 0
        # mock 학습곡선의 점근(상한) 능력치 — config가 좋을수록 높다.
        self._asymptote = _intrinsic_quality(cfg)

    def param_millions(self) -> float:
        """파라미터 수(M) — §6 메모리 예산/비용. real은 실측, mock은 근사."""
        if self.real_run is not None:
            return self.real_run.param_millions()
        d = int(self.cfg.get("d_model", 768))
        L = int(self.cfg.get("n_layers", 12))
        return 12 * d * d * L / 1e6

    def train(self, n_steps: int, rng: random.Random) -> None:
        """n_steps만큼 학습. real이면 실제 학습 루프."""
        if self.real_run is not None:
            self.real_run.train(n_steps, rng)
            self.trained_steps = self.real_run.trained_steps
            return
        self.trained_steps += n_steps

    # ── 평가 인터페이스(백엔드 공통) ─────────────────────────────────────────
    def proxy_score(self, task: str, rng: random.Random) -> float:
        """§5 프록시 점수(높을수록 우수). real=실측, mock=합성 신호."""
        if self.real_run is not None:
            return float(self.real_run.proxy_score(task, rng))
        if task == "factory_mqar":
            strength = self.memory_strength(rng)
            seg = int(self.cfg.get("segment_len", 256))
            difficulty = 1.0 - min(seg, 512) / 512 * 0.08
            return _clamp01(strength * difficulty)
        if task == "short_horizon_pred":
            return self.predict_skill(rng)
        raise ValueError(f"unknown proxy task: {task!r}")

    def full_score(self, rng: random.Random) -> float:
        """실제 과제 full 평가(상위 승급에만). real=어려운 평가, mock=합성."""
        if self.real_run is not None:
            return float(self.real_run.full_score(rng))
        recall = self.memory_strength(rng)
        pred = self.predict_skill(rng)
        return _clamp01(0.6 * recall + 0.4 * pred + rng.uniform(-0.06, 0.06))

    def _learning_curve(self) -> float:
        """현재까지 학습량 기준 실현 능력치 ∈ [0, asymptote]. 포화형 곡선."""
        # 16k 스텝쯤에서 점근치의 ~95%에 도달하도록 스케일.
        progress = 1.0 - math.exp(-self.trained_steps / 6000.0)
        return self._asymptote * progress

    def memory_strength(self, rng: random.Random) -> float:
        """recall 계열 능력치(유효 메모리 성장 가설). cfg의 기억 축에 민감."""
        base = self._learning_curve()
        noise = rng.uniform(-0.04, 0.04)
        return _clamp01(base * _recall_modifier(self.cfg) + noise)

    def predict_skill(self, rng: random.Random) -> float:
        """단기 예측 능력치. 표현 축(SSL)·base_rule에 민감."""
        base = self._learning_curve()
        noise = rng.uniform(-0.04, 0.04)
        return _clamp01(base * _predict_modifier(self.cfg) + noise)


def build(cfg: dict[str, Any]) -> ModelHandle:
    """§4 build(cfg) — 우리 torch 참조 구현(real) 우선, 없으면 mock 폴백.

    (업스트림 growing-memory-pytorch가 build API를 노출하면 그쪽을 우선 시도.)
    """
    mod = _try_import_real()
    if mod is not None:
        builder = getattr(mod, "build", None) or getattr(mod, "build_model", None)
        if callable(builder):
            try:
                return ModelHandle(cfg, backend="real-upstream", real_model=builder(cfg))
            except Exception:
                pass
    if _real_backend_available():
        try:
            from .real import RealRun
            return ModelHandle(cfg, backend="real", real_run=RealRun(cfg, REAL_RT))
        except Exception as e:  # noqa: BLE001
            import sys
            print(f"[model_adapter] real backend 실패 → mock 폴백: {e}", file=sys.stderr)
    return ModelHandle(cfg, backend="mock")


def backend_label() -> str:
    if _try_import_real() is not None:
        return "real-upstream"
    return "real" if _real_backend_available() else "mock"


def gpu_info() -> dict[str, Any]:
    """배포 노드의 GPU 가용성 프로브(§2 하드웨어). torch 있으면 CUDA/디바이스 보고.

    mock 백엔드라도 이 노드가 실제 4090을 인식하는지 확인하는 용도(GPU 파이프라인 검증).
    torch 미설치면 available=False로 비치명적 반환.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return {"torch": False, "cuda": False, "device": None}
    cuda = bool(torch.cuda.is_available())
    info: dict[str, Any] = {"torch": torch.__version__, "cuda": cuda, "device": None}
    if cuda:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["device"] = torch.cuda.get_device_name(idx)
        info["vram_gb"] = round(props.total_memory / 1e9, 1)
        # 실제 CUDA 연산 1회로 디바이스 도달 확인
        try:
            x = torch.randn(256, 256, device="cuda")
            _ = (x @ x).sum().item()
            info["cuda_op_ok"] = True
        except Exception as e:  # noqa: BLE001
            info["cuda_op_ok"] = False
            info["cuda_op_error"] = str(e)
    return info


# ── mock 능력치 모델 ─────────────────────────────────────────────────────────
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _intrinsic_quality(cfg: dict[str, Any]) -> float:
    """config의 '잠재 품질' 점근치 ∈ [0,1]. 합성이지만 그럴듯한 사전지식 반영.

    실물에선 학습이 이 값을 결정한다. mock에선 ASHA가 좋은 점을 고를 수 있도록
    설계 직관(긴 컨텍스트 recall엔 titans/ssc/grm + 충분한 segment_len 유리 등)을 부여.
    """
    q = 0.45
    q += {"linear": 0.00, "swla": 0.03, "dla": 0.05, "titans": 0.09}.get(
        cfg.get("base_rule"), 0.0
    )
    q += {"residual": 0.00, "soup": 0.02, "grm": 0.05, "ssc": 0.07}.get(
        cfg.get("aggregation"), 0.0
    )
    q += 0.02 if cfg.get("segmentation") == "logarithmic" else 0.0
    q += 0.02 if cfg.get("init_mode") == "checkpoint" else 0.0
    # 표현 축
    ssl = cfg.get("ssl")
    if ssl:
        q += {"vjepa": 0.05, "dinov2": 0.04, "vicreg": 0.02}.get(ssl.get("encoder"), 0.0)
        # positive_pair는 "가장 큰 레버"(설계 §3.2)
        q += {
            "consecutive_frame": 0.02,
            "same_unit_multiview": 0.05,
            "same_defect_class": 0.07,
        }.get(ssl.get("positive_pair"), 0.0)
    return _clamp01(q)


def _recall_modifier(cfg: dict[str, Any]) -> float:
    """recall 프록시에 대한 cfg 가중 — 기억 용량/세그먼트가 클수록 유리."""
    m = 1.0
    seg = int(cfg.get("segment_len", 256))
    m += min(seg, 512) / 512 * 0.15  # 긴 세그먼트 → recall 유리
    if cfg.get("aggregation") in ("ssc", "grm"):
        m += 0.05
    return m


def _predict_modifier(cfg: dict[str, Any]) -> float:
    """단기 예측 프록시에 대한 cfg 가중 — 표현 품질/base_rule에 민감."""
    m = 1.0
    if cfg.get("base_rule") in ("dla", "titans"):
        m += 0.05
    if cfg.get("ssl"):
        m += 0.05
    return m
