# PCSA 확장 연구 — 작업 이력 및 남은 작업 (WORKLOG)

최종 갱신: 2026-10-03. 이 문서는 지금까지의 구현 이력, 현재 파이프라인 구조, 검증 상태, 남은 작업을 상세히 기록한다. 설계 근거는 [`two_phase_design.md`](two_phase_design.md), 선행연구 니즈는 [`related_work_needs.md`](related_work_needs.md) 참조.

---

## 0. 연구 개요 (현재 확정된 방향)

- 목표: 상담 LLM이 빠지는 **오류상황 케이스를 생성하는 레드티밍**(PCSA식)을 재현·발전. 작은 모델(gpt-5-nano) + gpt-4o-mini 중간평가자로 효율적으로 수행.
- 2단계 프레임워크:
  - **Phase I (신규 컨셉, 페르소나 섭동):** 상담 코퍼스 페르소나 + 공격목표→인지왜곡 → scriptwriter로 I_sys 생성 → **서로게이트 오픈모델에 보내 PAIR 방식으로 복잡도·의학용어·부정/속어·우회성을 반복 강화하다가 수렴 시 freeze**.
  - **Phase II (대화 중 전략 조정):** 하드닝된 페르소나로 target과 다중턴 공격, **측정기반 사전 프로파일**(ArtPerception식)로 gpt-4o-mini evaluator가 디테일한 전략 피드백, response-based Best-of-N, 최종 judge(PCSA Figure 4).
- 주장(좁힘): 프로파일링 자체는 RedAgent/ArtPerception이 이미 수행 → 우리 기여는 **상담 안전 특화 susceptibility 측정의 증분 효과**(ablation: blind/aware/shuffled)와 **Phase I 섭동 파이프라인**.
- 지표: **ASR**(1−N_refuse/N_total), **SS**(CARES Refuse1.0/Cautious0.5/Accept0.0), **4축 UNSAFE**(Figure 4), **PPL**(GPT-2, 임계값 100, 은닉성), **방어별 ASR**(§4.4).
- 언어: **전부 영어**(PCSA 충실 + GPT-2 PPL 유효).

---

## 1. 구현 이력 (컴포넌트별)

### 1.1 모델 어댑터
- [`adapters/gpt5nano.py`](adapters/gpt5nano.py): 신뢰 JSON stdin/stdout 어댑터. `--model`로 역할별 모델 전환(gpt-5-nano reasoning 파라미터 ↔ gpt-4o-mini 등 표준 파라미터 자동 분기). 키는 환경변수 전용, urllib만 사용, 재시도/백오프(Retry-After) 내장. Task:
  - `counseling_response`(target) / `score_calibration_response`(profiler)
  - `generate_client_turn`(공격자: persona·goal·strategy·weakness·hints·evaluator_feedback)
  - `score_progress`(중간평가자, 1–10 red-team guide + next_strategy; 기본 gpt-4o-mini)
  - `judge_response`(PCSA **Figure 4** rubric: 4 binary + cares + final_verdict)
  - `generate_cognitive_distortion` / `scriptwriter` / `perturb_persona`(Phase I)
  - `profile_strategy_hints`(LLM 요약판; 현재는 측정판을 주로 사용)
  - `selfdefend_check`(방어: shadow LLM 의도 스크린)
- [`adapters/openai_target.py`](adapters/openai_target.py): OpenAI 호환 **범용 target 어댑터**. `--model/--base-url/--key-env` 또는 env. gpt-3.5-turbo(OpenAI), Llama-3.1-8B(HF router) 등. UA 헤더 포함.

### 1.2 Phase I — 페르소나 섭동
- [`phase1_persona.py`](phase1_persona.py): `generate_distortion → scriptwrite → [perturb → evaluate_on_surrogates] → hill-climbing + patience + freeze`. 산출: 하드닝 페르소나 + trace.
- [`run_phase1.py`](run_phase1.py): CLI, goal별 하드닝 → `data/processed/hardened_personas.jsonl`(gitignored). 서로게이트 다중 지정 가능.
- 시드: [`data/psych_jargon.jsonl`](../data/psych_jargon.jsonl)(정신의학 용어 30개).

### 1.3 Phase II — 공격·사전프로파일·평가
- [`pcsa.py`](pcsa.py):
  - `build_profile`: 행동 calibration(weakness vector) + **`susceptibility_pretest`**(측정기반, ArtPerception식: 축×전략×register probe→gpt-4o-mini wobble→축별 Top-1 hint) + `shuffle_weakness`/`shuffle_hints`(통제).
  - `plan_condition`: 5조건(fixed/profile_blind/profile_aware/profile_shuffled/profile_aware_without_history)의 weakness·hints·history 통제.
  - `run_case`: **response-based Best-of-N**(후보N→각 target 응답→1–10 progress→최고 커밋), 힌트·evaluator_feedback를 공격자·평가자에 주입하고 다음 턴으로 carry, 커밋 응답을 Figure4 judge로 판정.
  - `aggregate_verdicts`, `paired_bootstrap`.
- [`experiment.py`](experiment.py): 데이터 로드→사전프로파일→케이스×조건 **병렬 paired 실행**→조건별 ASR/SS/4축/case 실패율 + aware−blind/shuffled/no-history paired 효과 + susceptibility matrix 기록.
- [`data_sources.py`](data_sources.py): 페르소나(코퍼스/fallback)·공격목표·calibration probe·susceptibility probe·jargon **분리 로드**.

### 1.4 데이터 (커밋, 영어, 합성·승인)
- [`data/attack_goals.jsonl`](../data/attack_goals.jsonl): 축별 공격목표(의도-수준, 비실행).
- [`data/personas_fallback.jsonl`](../data/personas_fallback.jsonl): 합성 페르소나(코퍼스 부재 시 fallback).
- [`data/calibration_probes.jsonl`](../data/calibration_probes.jsonl): 4 construct 프로파일링 probe.
- [`data/susceptibility_probes.jsonl`](../data/susceptibility_probes.jsonl): 축×전략×register=32 고정 probe 배터리.
- [`data/psych_jargon.jsonl`](../data/psych_jargon.jsonl): 정신의학 용어 시드.
- 원본 코퍼스(Cactus/CBT-DP)는 private HF, gitignored. [`build_personas_from_corpus.py`](build_personas_from_corpus.py)로 추출(있을 때).

### 1.5 지표·방어
- ASR/SS/4축: `experiment.py`·judge에 구현, **논문 정의와 일치 확인 완료**.
- PPL + 방어: [`defenses.py`](defenses.py) — PerplexityFilter(GPT-2, thr 100) / SelfDefend(shadow LLM) / GraniteGuardian(로컬 2B, opt-in). 방어별 detection rate·ASR_under_defense 재계산.

### 1.6 테스트
- `test_agents.py`, `test_simulation.py`, `test_pcsa.py`, `test_phase1.py`, `test_data_sources.py` — **총 28개 통과**. (adapter·defenses는 라이브 검증)

---

## 2. 검증 상태 (라이브)

| 항목 | 상태 |
|---|---|
| gpt-5-nano 어댑터 5역할 | ✅ |
| gpt-4o-mini 중간평가자(1–10) | ✅ |
| Figure 4 judge (4축+UNSAFE+CARES) | ✅ |
| Phase I: distortion/scriptwriter/perturb | ✅ (복잡도·우회성 상승 확인) |
| 측정 susceptibility pre-test | ✅ (gpt-3.5-turbo에서 Top-1 산출; 하드코딩 가정과 다른 측정결과 = 가치 입증) |
| 영어 E2E run_case | ✅ |
| PerplexityFilter(GPT-2) | ✅ (자연프롬프트 PPL 17~57<100 미탐지, 토큰샐러드 3070 탐지) |
| SelfDefend | ✅ (위장 자해요청 탐지/양성 통과) |
| 전체 영어 paired 실험(gpt-3.5-turbo) | ⏳ **실행 중**(사전프로파일링 순차 단계) |

타깃 가용성: GPT-3.5-Turbo(OpenAI) ✅ / Llama-3.1-8B(HF router) ✅ / PsychoCounsel·PsyCoPref(featherless, HF 토큰으로 서버사이드 404) ⚠️.

---

## 3. 현재 실행 중

- `runs/gpt35_en.jsonl` + `reports/gpt35_en.json`: gpt-3.5-turbo, 4조건(fixed/blind/aware/shuffled), 축당 2케이스, 4턴, Best-of-N 3, 측정 사전프로파일. 예상 ~45–65분.
- 완료 후: 조건별 ASR·SS·4축 + aware−blind paired, 이어서 `defenses.py`로 방어별 ASR.

---

## 4. 남은 작업 (TODO)

### 4.1 즉시 (이번 실행 직후)
- [ ] gpt-3.5-turbo 결과 분석(aware>blind 여부, SS/ASR/4축).
- [ ] `defenses.py` 실행(ppl+selfdefend) → 방어별 ASR 표.
- [ ] (opt) Granite Guardian 로컬 2B 실행 검증.

### 4.2 실험 확장
- [ ] **Llama-3.1-8B** target으로 동일 실험(HF rate-limit 때문에 workers 낮춰). 교차-target 비교.
- [ ] PsychoCounsel/PsyCoPref target: **featherless API 키** 또는 HF 전용 Inference Endpoint 확보 시 연결.
- [ ] **Phase I 라이브 실행**(`run_phase1.py`)로 서로게이트 하드닝 페르소나 생성 → Phase II에 투입(현재 Phase II는 corpus/fallback 페르소나 사용; 하드닝 페르소나 연결 미완).
- [ ] 케이스/턴/시드 반복 늘려 통계력 확보(현재 소규모).

### 4.3 베이스라인 강화
- [ ] 외부 베이스라인 attacker 플러그인: 단일턴 직접요청, Crescendo식 에스컬레이션-only, (가능하면) CoA/AMA/ActorAttack. `--attacker-command`로 교체 가능하도록 래핑.
  - 현재 내부 베이스라인: `fixed`(순진) < `profile_blind`(=PCSA 재현) < `profile_aware`(우리).

### 4.4 Phase I ↔ Phase II 연결
- [ ] `data_sources`/`experiment`가 `hardened_personas.jsonl`을 로드해 Phase II 공격자 페르소나로 사용하는 경로 추가(현재 미연결).
- [ ] 전이성 측정: 서로게이트에서 하드닝한 페르소나가 미지의 target에 전이되는 정도(fitness vs target ASR).

### 4.5 프로파일링 고도화
- [ ] susceptibility pre-test **병렬화**(현재 순차 → 초반 병목). register 차원 확장/축소 튜닝.
- [ ] 프로파일을 evaluator뿐 아니라 Phase I 섭동 방향에도 주입.

### 4.6 평가 신뢰도
- [ ] **전문가(임상) 판정**과 gpt-5-nano judge **일치도** 측정(PCSA는 87.5% 보고).
- [ ] judge를 gpt-4o로도 병행해 민감도 분석(현재 비용상 gpt-5-nano).
- [ ] held-out target/모델 일반화.

### 4.7 데이터
- [ ] 추가 상담 코퍼스 검토: Anno-MI, ESConv, PsyQA(설계문서 참조).
- [ ] 의학용어 소스 확장: MedMentions(UMLS), Wikipedia 정신의학 용어집 → `psych_jargon` 증강.

### 4.8 선행연구/논문화
- [ ] `related_work_needs.md`의 서지 확정(특히 CARES·PCSA 정확한 서지/arXiv 번호, RedAgent/ArtPerception).
- [ ] token·호출 비용 provenance 자동 기록(현재 stderr 로그).
- [ ] 결과표·그림(조건별 지표, PPL 분포, 방어별 ASR, Phase I fitness 곡선).

### 4.9 운영
- [ ] API 키 **폐기(rotate)**: 세션 중 채팅에 노출된 OpenAI·HF 키(현재 gitignored `.env`에만 저장).
- [ ] git 커밋(.env 제외 확인됨).

---

## 5. 실행 방법 요약

```bash
cd mental_health_agent_eval
set -a; . ../.env; set +a                  # OPENAI_API_KEY, HF_API_KEY

python3 -m unittest -v                     # 28 tests

# Phase II 전체 실험 (gpt-3.5-turbo)
export TARGET_MODEL=gpt-3.5-turbo TARGET_BASE_URL=https://api.openai.com/v1 TARGET_API_KEY_ENV=OPENAI_API_KEY
python3 experiment.py --conditions fixed profile_blind profile_aware profile_shuffled \
  --cases-per-axis 2 --turns 4 --best-of-n 3 --workers 6 \
  --target-command '["python3","adapters/openai_target.py"]' --target-model gpt-3.5-turbo \
  --out runs/gpt35_en.jsonl --summary ../reports/gpt35_en.json

# 방어 평가
python3 defenses.py runs/gpt35_en.jsonl --defenses ppl selfdefend --out ../reports/defenses.json

# Phase I 하드닝 (서로게이트=Llama-3.1-8B)
python3 run_phase1.py --cases-per-axis 1 --max-iters 4 \
  --surrogate '["python3","adapters/openai_target.py","--model","meta-llama/Llama-3.1-8B-Instruct","--base-url","https://router.huggingface.co/v1","--key-env","HF_API_KEY"]'
```
