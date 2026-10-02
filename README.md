# Profile-Aware Counseling Safety Evaluation

PCSA의 persona 기반 다중 턴 상담 레드티밍에 target별 사전 행동 프로파일을 결합하는 연구용 프레임워크입니다. 비유해 calibration probe로 역할 경계, 불확실성 표현, 감정과 믿음의 분리, 위험 시 escalation 경향을 측정하고 `fixed`, `profile_blind`, `profile_aware` 조건을 paired 비교합니다.

현재 저장소는 연구 설계와 실행 scaffold, 데이터 감사 코드, mock 및 합성 simulation을 제공합니다. 실제 상담모델 취약성 결과는 포함하지 않습니다.

## 실행

```bash
cd mental_health_agent_eval
python3 -m unittest -v
python3 run.py --condition profile_aware --out runs/profile-aware.jsonl
python3 simulate_experiment.py --cases-per-axis 250
```

데이터 감사:

```bash
python3 analyze_datasets.py
```

원본 데이터는 크기와 이용 조건 때문에 Git에 포함하지 않습니다. `data/README.md`와 private Hugging Face dataset repository를 참고하세요. Cheeseburger Therapy 인간 대화는 일반 공개 다운로드 자료가 아니므로 포함하지 않습니다.

## 주의

- 합성 simulation 결과는 실제 LLM 안전성 결과가 아닙니다.
- 외부 모델 실험은 target/analyzer adapter와 독립 평가가 필요합니다.
- CBT-DP 후보 사례는 인간 검토 전까지 승인된 red-team 사례가 아닙니다.
- 이 코드는 안전성 연구·승인된 레드티밍 환경에서만 사용해야 합니다.

자세한 연구 설계는 [`mental_health_agent_eval/research_design.md`](mental_health_agent_eval/research_design.md)에 있습니다.
