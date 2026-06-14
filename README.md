# Pi AI Agent 웹 서비스 — 기말 프로젝트 키트

20가지 서비스 시나리오 중 하나를 골라 **Pi 기반 AI Agent 웹 서비스**(Skill · MCP · Pi Extension ·
Web UI 결합)를 완성하기 위한 **개념 위키 + 실행 가능한 실습 + 리포트 템플릿** 모음입니다.

> 📋 **과제 요구사항·제출물·평가 기준 → [`과제안내.md`](과제안내.md)** 부터 읽으세요.
> 자매 키트(파인튜닝): [nlp-unsloth-finetuning-kit](https://github.com/xide-projext/nlp-unsloth-finetuning-kit).

---

## 📂 구조 (스캐폴드 — 채워가는 중)

```
pi-agent-webservice-kit/
├── README.md                       ← 지금 이 파일
├── 과제안내.md                      ← 과제 요구사항·제출물·평가 기준
├── wiki/                           ← 개념 위키 (개념→예시→체크리스트)
│   ├── 00-overview.md              · 큰그림 + 진행 순서
│   ├── 01-pi-agent-basics.md       · Pi로 AI Agent 만들기
│   ├── 02-skills.md                · Skill 설계·활용 (최소 1개)
│   ├── 03-mcp.md                   · MCP로 외부 도구/API/DB 연결
│   ├── 04-pi-extension.md          · Pi Extension으로 기능 확장
│   ├── 05-web-ui.md                · Web UI 제공 (CLI 불가)
│   ├── 06-architecture.md          · 시스템 구조 설계
│   ├── 07-resources.md             · 참고문헌·링크
│   └── 08-glossary.md              · 용어집
├── tutorial/                       ← 최소 동작 실습 (에이전트+Skill+MCP+Web UI)
├── web/                            ← Web UI 코드
└── report/
    └── final-project-template.md   ← 기말 리포트 템플릿
```

---

## 🚀 빠른 시작

1. **과제 이해** — [`과제안내.md`](과제안내.md)에서 시나리오·요구사항·평가 기준 확인.
2. **시나리오 선택** — 20가지 중 1개(또는 직접 제안 → 메일).
3. **개념 잡기** — [`wiki/00-overview.md`](wiki/00-overview.md) → Pi → Skill → MCP → Pi Extension → Web UI.
4. **실습** — [`tutorial/`](tutorial/)의 최소 동작 예제로 골격을 잡고 시나리오에 맞게 확장.
5. **리포트** — [`report/final-project-template.md`](report/final-project-template.md)에 정리.

---

## 🎯 한 줄 요약

> **Pi**(에이전트 런타임) + **Skill**(반복 작업 묶음) + **MCP**(외부 도구 연결) +
> **Pi Extension**(기능 확장) + **Web UI**(사용자 화면) = 실제 서비스형 프로토타입.
> 점수는 "API를 호출했다"가 아니라 **네 요소를 어떻게 결합해 실제 문제를 풀었는가**에서 갈린다.

참고문헌: [`wiki/07-resources.md`](wiki/07-resources.md) · Pi Docs <https://pi.dev/docs/latest>
