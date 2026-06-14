"""§3.4 유효성 게이트 — 학습 전, 작은 차원으로 동치/shape 테스트를 통과시킨다.

샘플한 config가 growing-memory-pytorch에서 "정의상 유효"한지 학습 비용을 들이기
전에 거른다. 실물 패키지가 있으면 그쪽의 동치/shape 테스트를 작은 차원으로 호출하고,
없으면 설계상의 제약(논문 충실성 규칙)을 코드로 검사하는 mock 게이트를 쓴다.

게이트를 통과한 config만 train_and_eval로 진입한다(논문 충실성·유효성 보장).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_adapter import try_equivalence_test


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


# 설계상의 정합성 규칙(mock 게이트). 실물 패키지의 동치 테스트가 없을 때 사용.
_VALID_BASE_RULE = {"linear", "swla", "dla", "titans"}
_VALID_AGG = {"residual", "grm", "soup", "ssc"}
_VALID_SEG = {"constant", "logarithmic"}
_VALID_INIT = {"checkpoint", "independent"}


def _structural_checks(cfg: dict[str, Any]) -> GateResult:
    """차원/조합 정합성 검사 — shape 불가능·무의미 조합을 차단."""
    if cfg.get("base_rule") not in _VALID_BASE_RULE:
        return GateResult(False, f"unknown base_rule={cfg.get('base_rule')}")
    if cfg.get("aggregation") not in _VALID_AGG:
        return GateResult(False, f"unknown aggregation={cfg.get('aggregation')}")
    if cfg.get("segmentation") not in _VALID_SEG:
        return GateResult(False, f"unknown segmentation={cfg.get('segmentation')}")
    if cfg.get("init_mode") not in _VALID_INIT:
        return GateResult(False, f"unknown init_mode={cfg.get('init_mode')}")

    d_model = int(cfg.get("d_model", 0))
    n_heads = int(cfg.get("n_heads", 0))
    if d_model <= 0 or n_heads <= 0:
        return GateResult(False, "d_model/n_heads must be positive")
    if d_model % n_heads != 0:  # head 차원이 정수여야 함(shape 게이트)
        return GateResult(False, f"d_model({d_model}) % n_heads({n_heads}) != 0")

    seg_len = int(cfg.get("segment_len", 0))
    if seg_len <= 0:
        return GateResult(False, "segment_len must be positive")

    # top_k는 ssc 전용 — 다른 aggregation에 붙어 있으면 무효(논문 충실성)
    if "top_k" in cfg and cfg.get("aggregation") != "ssc":
        return GateResult(False, "top_k is only valid for aggregation=ssc")
    if cfg.get("aggregation") == "ssc":
        top_k = int(cfg.get("top_k", 0))
        if top_k <= 0:
            return GateResult(False, "ssc requires positive top_k")
        # top_k 슬롯이 세그먼트보다 많으면 의미 없음
        if seg_len > 0 and top_k > max(2, seg_len // 8):
            return GateResult(False, f"top_k({top_k}) too large for segment_len({seg_len})")

    # checkpoint 초기화는 표현축이 정의돼 있어야 의미 있음(인코더 출력 캐시에서 init)
    if cfg.get("init_mode") == "checkpoint" and "ssl" in cfg:
        enc = cfg["ssl"].get("encoder")
        if not enc:
            return GateResult(False, "init_mode=checkpoint requires ssl.encoder")

    return GateResult(True, "structural ok")


def validity_gate(cfg: dict[str, Any]) -> GateResult:
    """학습 전 유효성 게이트. 통과 시 GateResult(ok=True).

    1) 구조/shape 검사(항상)
    2) 실물 growing-memory-pytorch 동치/shape 테스트를 *작은 차원*으로 시도
       (패키지 없으면 자동 skip).
    """
    structural = _structural_checks(cfg)
    if not structural.ok:
        return structural

    eq = try_equivalence_test(cfg)  # (ran: bool, ok: bool, reason: str)
    if eq.ran and not eq.ok:
        return GateResult(False, f"equivalence test failed: {eq.reason}")

    detail = "equivalence ok" if eq.ran else "structural ok (equivalence skipped: no growing-memory)"
    return GateResult(True, detail)
