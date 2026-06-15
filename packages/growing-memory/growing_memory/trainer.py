"""실제 학습/평가 — AdamW + masked CE + SSL aux. RealRun이 모델·옵티마이저·평가를 보유.

ASHA 승급 시 같은 RealRun을 계속 학습(누적 스텝)하므로 rung 간 학습이 이어진다.
proxy_score: MQAR recall(연관회상) 또는 short_horizon의 1/(1+CE). full_score: 더 어려운 평가.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .data import make_mqar_batch, make_short_horizon_batch
from .model import build_real


def _ce(logits, tgt):
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)


class RealRun:
    """실물 모델 1개의 학습/평가 핸들."""

    def __init__(self, cfg: dict, rt: dict):
        self.cfg = cfg
        self.rt = rt
        dev = rt.get("device", "cuda")
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        self.task = rt["task"]
        self.vocab = rt["vocab"]
        self.seq_len = rt["seq_len"]
        self.num_pairs = rt["num_pairs"]
        self.batch = rt["batch"]
        self.trained_steps = 0
        self.backend = "real"

        max_len = max(rt["seq_len"], rt.get("full_seq_len", rt["seq_len"])) + 2
        self.model = build_real(cfg, self.vocab, max_len).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=rt.get("lr", 3e-3),
                                     weight_decay=0.01)
        self.train_gen = torch.Generator().manual_seed(rt.get("seed", 0))
        self.eval_gen = torch.Generator().manual_seed(12345)  # 평가 고정

    def param_millions(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def _batch(self, train: bool, hard: bool = False):
        gen = self.train_gen if train else self.eval_gen
        if self.task == "short_horizon_pred":
            L = self.seq_len if not hard else int(self.seq_len * 1.5)
            return make_short_horizon_batch(self.batch, L, self.vocab, self.device, gen)
        pairs = self.num_pairs if not hard else int(self.num_pairs * 1.5)
        L = self.seq_len if not hard else int(self.seq_len * 1.5)
        return make_mqar_batch(self.batch, L, pairs, self.vocab, self.device, gen)

    def train(self, n_steps: int, rng=None) -> None:
        self.model.train()
        for _ in range(max(0, n_steps)):
            inp, tgt, _ = self._batch(train=True)
            logits, aux = self.model(inp, return_aux=True)
            loss = _ce(logits, tgt) + aux
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.trained_steps += 1

    def _infer(self, inp):
        """추론 경로 — eval_recurrent=True면 상수메모리 RNN 재귀(엣지 배포 경로와 동일).

        병렬형과 동치(diff ~1e-7)지만 L×L 어텐션을 만들지 않아 메모리가 길이 무관.
        swla는 forward_recurrent 내부에서 병렬로 위임된다.
        """
        if self.rt.get("eval_recurrent", True):
            return self.model.forward_recurrent(inp)
        return self.model(inp)

    @torch.no_grad()
    def _eval(self, hard: bool, n_batches: int = 4):
        self.model.eval()
        self.eval_gen.manual_seed(12345)     # 평가 배치 고정(재현성·proxy↔full 공정 비교)
        correct = total = 0
        ce_sum = ce_n = 0.0
        for _ in range(n_batches):
            inp, tgt, qmask = self._batch(train=False, hard=hard)
            logits = self._infer(inp)
            ce_sum += _ce(logits, tgt).item()
            ce_n += 1
            if qmask.any():
                pred = logits.argmax(-1)
                sel = qmask
                correct += (pred[sel] == tgt[sel]).sum().item()
                total += sel.sum().item()
        recall = (correct / total) if total else 0.0
        ce = ce_sum / max(1, ce_n)
        return recall, ce

    def proxy_score(self, task: str, rng=None) -> float:
        recall, ce = self._eval(hard=False)
        if task == "short_horizon_pred":
            return 1.0 / (1.0 + ce)          # CE 낮을수록 높은 점수
        return recall                         # factory_mqar: 회상 정확도

    def full_score(self, rng=None) -> float:
        recall, ce = self._eval(hard=True, n_batches=6)
        if self.task == "short_horizon_pred":
            return 1.0 / (1.0 + ce)
        return recall


def train_and_eval(cfg: dict, rt: dict, steps: int) -> dict:
    """단발 학습+평가(스모크용)."""
    run = RealRun(cfg, rt)
    run.train(steps)
    recall, ce = run._eval(hard=False)
    return {"params_M": round(run.param_millions(), 2), "recall": round(recall, 4),
            "val_ce": round(ce, 4), "val_bpb": round(ce / math.log(2), 4),
            "steps": run.trained_steps}
