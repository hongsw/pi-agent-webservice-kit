"""실제 학습 데이터 — MQAR(연관회상)과 short-horizon(다음토큰) 토큰열 생성.

MQAR: 긴 컨텍스트 안에 (key,value) 쌍을 흩뿌리고 끝에서 key로 value를 회상시킨다.
      growing-memory의 핵심 가설(유효 메모리 성장 → recall)을 직접 측정하는 실 과제.
short_horizon: 잡음 섞인 토큰열의 다음 토큰 예측(자기지도). val CE/bpb로 평가.

모두 torch 텐서를 즉석 생성(라벨 불필요/합성이지만 실제 학습 신호). ignore_index=-100.
"""

from __future__ import annotations

import torch


def make_mqar_batch(
    batch: int,
    seq_len: int,
    num_pairs: int,
    vocab: int,
    device: str,
    gen: torch.Generator,
    num_queries: int | None = None,
):
    """MQAR 배치 생성.

    레이아웃(샘플별): [k1,v1,...,kN,vN, (filler...), q1,q2,...,qM]
      - 회상 대상 query 위치의 target = 해당 key의 value, 그 외 -100.
    토큰 분할: 0=pad, 키 토큰군과 값 토큰군을 분리해 모호성 제거.
    반환: input_ids[B,L], targets[B,L], query_mask[B,L](평가용 bool)
    """
    if num_queries is None:
        num_queries = num_pairs
    # 키/값 토큰 영역 분리(겹치지 않게). pad=0.
    half = max(2, (vocab - 1) // 2)
    key_lo, key_hi = 1, 1 + half           # [1, half]
    val_lo, val_hi = 1 + half, vocab       # [half+1, vocab)

    # 최소 길이 보장(2N kv + M query). 남는 길이는 kv와 query 사이 distractor로 채워
    # '긴 컨텍스트 회상' 난이도를 준다(메모리 가설 측정). distractor는 값/키와 겹치지 않는 0(pad).
    n_kv = 2 * num_pairs
    L = max(seq_len, n_kv + num_queries)
    inp = torch.zeros(batch, L, dtype=torch.long)
    tgt = torch.full((batch, L), -100, dtype=torch.long)
    n_q = num_queries

    for b in range(batch):
        keys = torch.randperm(key_hi - key_lo, generator=gen)[:num_pairs] + key_lo
        vals = torch.randint(val_lo, val_hi, (num_pairs,), generator=gen)
        kv = torch.empty(n_kv, dtype=torch.long)
        kv[0::2] = keys
        kv[1::2] = vals
        inp[b, :n_kv] = kv                      # 앞쪽: kv 쌍
        qidx = torch.randperm(num_pairs, generator=gen)[:n_q]
        qstart = L - n_q                        # 뒤쪽: query (그 사이는 pad distractor)
        inp[b, qstart:] = keys[qidx]
        tgt[b, qstart:] = vals[qidx]

    inp = inp.to(device)
    tgt = tgt.to(device)
    qmask = tgt.ne(-100)
    return inp, tgt, qmask


def make_short_horizon_batch(
    batch: int,
    seq_len: int,
    vocab: int,
    device: str,
    gen: torch.Generator,
):
    """다음 토큰 예측(자기지도). 약한 구조(주기적 반복 + 잡음)로 학습 신호 부여.

    토큰열에 주기 p의 반복 패턴을 심어 모델이 단기 의존성을 학습하면 CE가 내려간다.
    반환: input_ids[B,L], targets[B,L](shift), val CE 평가는 trainer가.
    """
    L = seq_len + 1
    seq = torch.randint(1, vocab, (batch, L), generator=gen)
    # 주기 패턴: 일부 위치를 p 이전 토큰으로 복사(예측 가능 신호)
    p = 4
    for t in range(p, L):
        mask = (torch.rand(batch, generator=gen) < 0.5)
        seq[mask, t] = seq[mask, t - p]
    inp = seq[:, :-1].contiguous().to(device)
    tgt = seq[:, 1:].contiguous().to(device)
    return inp, tgt, tgt.ne(-100)
