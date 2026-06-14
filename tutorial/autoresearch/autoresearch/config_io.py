"""run config 로더 — PyYAML이 있으면 사용, 없으면 우리 스키마(§9)에 맞춘 미니 파서.

지원: 들여쓰기 기반 중첩 맵, `key: value`, 인라인 리스트 `[a, b, c]`, `# 주석`, 스칼라
(int/float/bool/str). 블록 시퀀스(`- item`)는 쓰지 않으므로 미지원(스키마가 인라인 리스트만 사용).
"""

from __future__ import annotations

from typing import Any


def load_run_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return _mini_yaml(text)


def _scalar(tok: str) -> Any:
    tok = tok.strip()
    if tok == "" or tok == "~" or tok.lower() == "null":
        return None
    if tok.lower() in ("true", "false"):
        return tok.lower() == "true"
    if (tok[0] == tok[-1]) and tok[0] in ("'", '"') and len(tok) >= 2:
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def _parse_value(val: str) -> Any:
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_scalar(x) for x in _split_top(inner)]
    return _scalar(val)


def _split_top(s: str) -> list[str]:
    """쉼표 분리(대괄호 중첩은 스키마에 없어 단순 분리)."""
    return [p for p in (x.strip() for x in s.split(",")) if p != ""]


def _strip_comment(line: str) -> str:
    out, in_str, q = [], False, ""
    for ch in line:
        if in_str:
            out.append(ch)
            if ch == q:
                in_str = False
        elif ch in ("'", '"'):
            in_str = True
            q = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _mini_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    # 스택: (indent, container)
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            continue
        key, _, val = content.partition(":")
        key = key.strip()
        # 현재 들여쓰기보다 깊은 스택은 팝
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            # 중첩 맵 시작
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_value(val)
    return root
