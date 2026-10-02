# profile_aware 설계를 위한 선행연구 니즈 정리

이 문서는 우리 방법(작은 모델 gpt-5-nano로 PCSA식 공격을 수행하되, 공격 전에 측정한 **target 행동 프로파일 prior**를 사용하는 `profile_aware`)의 각 설계결정을 어떤 선행연구로 정당화/차별화해야 하는지 매핑한다. 각 항목에 (a) 필요한 이유(우리 코드의 어느 부분), (b) 찾아야 할 논문 유형, (c) 내가 아는 유력 후보(정확한 서지는 확인 필요), (d) 확인/수집 포인트를 적는다.

우리 측정 construct 4개: `role_boundary`, `uncertainty_calibration`, `emotion_belief_separation`, `risk_escalation` (`agents.py`, `data/calibration_probes.jsonl`).
우리 축 4개(PCSA): `harmful_content`, `impersonation_violation`, `toxic_empathy`, `target_compliance` (`pcsa.py`).

---

## A. 주장(좁힌 버전): "분리된 상담 행동 측정치의 '추가 기여' 검증"

> ⚠️ 주의(2026-10-03 반영): "사전 프로파일을 처음 사용한다"는 **과장**이다. **RedAgent**(공격 전 target의 용도·기능을 프로파일링)와 **ArtPerception**(모델별 사전 측정 활용)이 이미 target 프로파일링을 한다. 따라서 주장을 **"분리된 상담 행동 측정치(behavioral measurement)의 추가 기여를 ablation으로 검증한다"**로 좁힌다. `profile_aware` vs `profile_blind` vs `profile_shuffled`의 paired 비교(`experiment.py`)가 그 검증 장치다. 이것은 **헤드라인이 아니라 보조 ablation**이며, 본 연구의 포커스는 아래 A0(상담 오류상황 케이스 생성 레드티밍의 발전)이다.

### A0. (포커스) 상담 맥락 오류상황 케이스 생성 레드티밍 — PCSA의 발전
- 필요 이유: PCSA는 "모델 특화 공격"이 아니라 **상담 LLM이 빠지는 오류상황(사례)을 persona 기반으로 생성**하는 레드티밍이다. 우리의 기여는 이 케이스 생성을 **더 발전**시키는 것: (1) 페르소나(코퍼스)·공격목표(분리된 세트)·다중턴 에스컬레이션·response-based Best-of-N(1–10 progress)·Fig4 judge의 체계화, (2) 전 역할을 작은 모델(gpt-5-nano)로 수행하는 **효율성**, (3) 4축 전반의 체계적 커버리지.
- 찾을 것: "무엇을 더 발전시켰다"를 주장하려면 PCSA 및 유사 상담 레드티밍의 **한계**(단일 전략, 평가 자동화의 편향, 축 커버리지, 비용)를 명시한 문장이 필요. PCSA 원문의 한계/future work 절, 그리고 경쟁 방법(CoA, AMA, Crescendo, ActorAttack — 이미지 자료에 등장)의 서지.
- 확인 포인트: CoA / AMA / ActorAttack 정확한 서지(우리 비교 baseline 후보), PCSA가 명시한 한계.

### 적응형 레드티밍·target 프로파일링 (우리 보조 ablation의 선행)

다음 세 흐름과의 **차별점**(= 우리는 "상담 행동 측정의 분리된 추가 기여"만 주장)을 명확히 해야 한다.

- **RedAgent** — 공격 전 target의 용도·기능 프로파일링. (서지 확인) → "프로파일링 자체는 신규가 아님"의 핵심 인용.
- **ArtPerception** — 모델별 사전 측정 활용. (서지 확인) → 동일.

### A1. 적응형/반복 최적화 공격 (프로파일 없이 target 응답에 적응)
- 필요 이유: `profile_blind`가 이미 Best-of-N으로 응답에 적응한다. 우리는 "응답 적응"이 아니라 "**사전 측정한 행동 prior**"가 추가 이득을 준다고 주장. 이 계열이 우리의 직접 baseline/비교군.
- 찾을 것: 블랙박스 LLM 자동 jailbreak의 반복 최적화·트리탐색·평가자 피드백 공격.
- 유력 후보(서지 확인):
  - PAIR — Chao et al., 2023, *Jailbreaking Black Box LLMs in Twenty Queries*.
  - TAP — Mehrotra et al., 2023, *Tree of Attacks with Pruning*.
  - GCG — Zou et al., 2023, *Universal and Transferable Adversarial Attacks on Aligned LMs*.
  - AutoDAN — Liu et al., 2023.
- 확인 포인트: 이들이 **target별 사전 프로파일을 분리된 calibration set에서 만들고 그걸 공격에 투입**하지 않는다는 점(= 우리 신규성)을 문장으로 못박을 근거.

### A2. 다중 턴 점진 공격 (에스컬레이션)
- 필요 이유: 우리 공격은 다중 턴 + 턴별 에스컬레이션(`adapters/gpt5nano.py`의 attacker escalation). PCSA와 공통. 차별점은 prior.
- 유력 후보: Crescendo — Russinovich et al., 2024; Many-shot jailbreaking — Anil et al., 2024.
- 확인 포인트: 다중 턴 효과를 별도로 통제해야 한다는 근거(우리 `profile_aware_without_history` 조건의 정당화).

### A3. 공격 prior의 전이성(transferability)·target-specific vs target-agnostic
- 필요 이유: 우리 `profile_shuffled` 조건(다른 target의 프로파일을 주면 효과가 사라지는가)이 "target-specific 정보 기여"를 증명. 이 ablation의 선행 근거.
- 찾을 것: adversarial prompt/suffix의 모델 간 전이, target-specific 최적화 이득 측정.
- 유력 후보: GCG의 transfer 실험(Zou 2023), 그리고 transfer 한계를 다룬 후속.
- 확인 포인트: "target-specific 최적화가 transfer보다 낫다"는 정량 결과가 있는 논문(우리 shuffled<aware 결과를 맥락화).

---

## B. "행동 프로파일" 개념 자체 (LLM을 probe로 특성화)

### B1. LLM 심리측정/성격·성향 프로파일링
- 필요 이유: "프로파일"이 숨은 성격 추론이 아니라 **관찰 가능한 반응 경향의 점수**라는 우리 정의(`research_design.md`)를 선행연구로 받쳐야 함.
- 유력 후보:
  - Safdari et al., 2023, *Personality Traits in Large Language Models*.
  - Jiang et al., 2023, *Evaluating and Inducing Personality in Pre-trained LMs*.
  - Perez et al., 2022, *Discovering Language Model Behaviors with Model-Written Evaluations* (행동을 probe로 측정).
- 확인 포인트: probe 기반으로 모델 **행동 경향을 재현 가능하게 수치화**한 방법론·신뢰도.

### B2. Uncertainty calibration (우리 construct: uncertainty_calibration)
- 필요 이유: profiler가 "불확실성 표현"을 점수화. 이 construct의 측정 타당성 근거.
- 유력 후보: Kadavath et al., 2022, *Language Models (Mostly) Know What They Know*; Tian et al., 2023, *Just Ask for Calibration*.
- 확인 포인트: 자기보고식 확신/calibration 측정의 방법과 한계.

### B3. Sycophancy / 유해 믿음 승인 (우리 construct: emotion_belief_separation, 축: toxic_empathy)
- 필요 이유: toxic_empathy = 감정 공감이 **해로운 믿음의 승인**으로 넘어가는 실패. 핵심 축.
- 유력 후보:
  - Sharma et al., 2023, *Towards Understanding Sycophancy in Language Models* (Anthropic).
  - Perez et al., 2022 (sycophancy eval 포함).
- 확인 포인트: sycophancy를 **공감 vs 사실 동의 분리**로 조작/측정한 선행(우리 emotion_belief 분리 probe의 근거). 상담 맥락 sycophancy가 있으면 최우선.

### B4. 역할 경계·과도한 권위주장 (construct: role_boundary, 축: impersonation_violation)
- 필요 이유: 모델이 임상 자격을 사칭/진단 단정하는 실패.
- 찾을 것: LLM의 전문직 역할 사칭, 의료 면책/overclaiming, medical advice 경계.
- 확인 포인트: 의료/상담 LLM의 자격·진단 경계 위반을 정의·측정한 연구.

---

## C. 레드티밍 일반 프레임·자동화

- 필요 이유: 우리 파이프라인(profiler/attacker/evaluator/judge 분리, 정보 격리)의 방법론적 위치.
- 유력 후보:
  - Perez et al., 2022, *Red Teaming Language Models with Language Models*.
  - Ganguli et al., 2022, *Red Teaming LMs to Reduce Harms* (Anthropic).
  - Best-of-N / 평가자-가이드 생성 선택(우리 selector 근거): rejection sampling / Best-of-N 관련.
- 확인 포인트: 공격자·평가자·judge를 **분리하고 test judgment을 공격에 되먹이지 않는** 설계의 선행(우리 정보격리 설계의 근거).

---

## D. 정신건강 LLM 안전성 평가 + 지표

### D1. CARES / 상담 안전 벤치마크 + 지표(ASR, SS)
- 필요 이유: 우리 지표 ASR(1−N_refuse/N_total), SS(Refuse1/Cautious.5/Accept0)가 **CARES 정의 그대로**. 원출처 서지 필수.
- 찾을 것: CARES 벤치마크 원논문(ASR·SS·Refuse/Cautious/Accept 정의), 그리고 PCSA가 인용한 판정 rubric.
- **최우선**: CARES 정확한 서지와 SS/ASR 수식 원문.

### D2. PCSA 및 상담 persona 시뮬레이션 공격
- 필요 이유: 우리의 직접 비교 대상(baseline=우리 blind가 PCSA 재현).
- 확인 포인트: Xu et al., EMNLP 2026, *Do No Harm ... PCSA*의 정확한 서지/arXiv 번호(현재 2604.04842로 참조 중 — 번호 검증 필요), 4전략·Best-of-N·judge 모델(GPT-4o-mini 평가자, GPT-4o judge) 수치.

### D3. 상담 LLM 안전 taxonomy / 유해 공감·과의존
- 찾을 것: 정신건강 챗봇의 위해 유형 분류, 위기(자살) 대응 누락, over-reliance.
- 확인 포인트: 우리 4축(특히 toxic_empathy, harmful_content)과 매핑되는 임상적 위해 taxonomy. crisis 누락을 별도 축으로 둘지 결정에 필요.

### D4. LLM-as-judge 신뢰도
- 필요 이유: 우리 judge가 gpt-5-nano. 자동 판정 타당성·사람 일치도 논거(PCSA는 87.5% 보고).
- 유력 후보: Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*.
- 확인 포인트: 안전 판정에서 judge 편향·사람 일치도 측정 방법.

---

## E. 데이터셋 원출처 (서지/라이선스 확정)

- Cactus — 합성 상담 대화 (repo: coding-groot/cactus). 원논문 서지.
- CBT-Bench / CBT-DP — mianzhang/CBT-Bench. 원논문 서지·라이선스.
- Cheeseburger Therapy — CHI 2024 상담 대화 116세션. 원논문 서지(IRB/consent 근거).
- PsychoCounsel-Preference / PsychoCounsel-Llama3-8B — arXiv 2502.19731, *Preference Learning Unlocks LLMs' Psycho-Counseling Skills*. (우리 target 후보)
- 확인 포인트: 각 데이터의 라이선스와 "페르소나 구성 vs 공격 목표 분리"에 쓰는 용례의 정당성.

---

## 우선순위 (당신이 먼저 찾아주면 좋은 순서)

1. **CARES 원논문** (ASR/SS 정의) + **PCSA 정확한 서지/arXiv 번호 검증**. — 지표·baseline 확정에 필수.
2. **A1 적응형 공격(PAIR/TAP/GCG)** + **A3 transferability**. — profile_aware 신규성과 shuffled ablation의 핵심.
3. **B3 sycophancy(Sharma 2023)** + 상담 맥락 sycophancy. — toxic_empathy 축의 이론적 근거.
4. **A2 다중 턴(Crescendo/Many-shot)**. — history 조건 정당화.
5. B1 심리측정, B2 calibration, C 레드티밍 프레임, D4 LLM-judge. — 방법 일반 근거.
6. B4 역할경계, D3 상담 taxonomy, E 데이터 서지. — 보강.

각 논문에 대해 (제목·저자·연도·링크 + 한두 줄 핵심 + 우리 어느 항목에 쓰는지)만 정리해 주면, Related Work와 Method 정당화에 바로 반영하겠습니다.
