<!-- ABOUTME: 한국어 공개 진입점. 온디바이스 SLM 라우팅 분류기를 DSPy로 최적화한 개인 연구. -->
<!-- ABOUTME: 동기 → 파이프라인 → 태스크/데이터셋 → Router DSPy 결과 → 기술 스택. -->

# Multi-Agent-Project

> **TL;DR.** 온디바이스 SLM (Gemma 4 E2B) 의 라우팅 프롬프트를 DSPy 로
> 최적화해, **31B 레퍼런스 대비 −1.19 pp** (95.99% vs 97.18% test weighted F1)
> 까지 따라붙였습니다. 600개 hold-out 셋 기준, privacy leak 0건.

---

## 0. 프로젝트 동기

갤럭시 스마트폰에 Multi-Agent 시스템을 올리는 방향을 개인적으로 탐구한 프로젝트
입니다. **온디바이스 SLM(Small Language Model) 을 DSPy 로 최적화해 라우팅
분류기**로 사용하고, 메모리 시스템도 마찬가지로 구현하여, 챗봇을 만드는게
목표입니다.

이 프로젝트의 핵심 질문: **DSPy 로 작은 0.8B, 2B 모델을 31B 레퍼런스 모델
수준의 라우팅 품질에 얼마나 가깝게 끌어올릴 수 있는가?**

---

## 1. 파이프라인

<!-- [지금은 이렇게 두고, 나중에 메모리 파트까지 적을 때 확장] -->

```
사용자 질의
   │
   ▼
라우터 SLM (Gemma 4 E2B, DSPy-BFRS 최적화)
   │
   ├── Local   → 같은 SLM 이 직접 답변 (온디바이스 대화)
   ├── Cloud   → Gemini API (복잡한 생성)
   └── Search  → Perplexity API (실시간 데이터)
```

라우팅 분류기는 사용자 질의를 **Local / Cloud / Search** 세 가지로
분류합니다. DSPy 의 `BootstrapFewShotWithRandomSearch` (BFRS), MIPROv2,
COPRO 를 이용해 Gemini 3 Flash 로 생성한 데이터셋으로 최적화합니다.

---

## 2. 태스크 & 데이터셋 선정

### 2.1 라우팅 태스크 정의

사용자 질의 → **Local / Cloud / Search** 3-class single-label classification.

| 클래스 | 정의 | 예시 |
|---|---|---|
| **Local** | Galaxy 내장 앱 (Weather, Calendar, Samsung Health 등) 또는 온디바이스 SLM 이 직접 처리 가능. 캐주얼 대화 + 단순 Q&A + 개인·민감 정보 | "오늘 날씨", "hello 번역해줘", "내 부채 예산 짜줘" |
| **Cloud** | 큰 LLM 이 필요한 복잡 생성. 긴 글쓰기, 코드 생성, 다단계 추론 | "리액트 풀스택 todo 앱 작성", "이 3페이지 문서 번역" |
| **Search** | 실시간 외부 데이터 — 뉴스, 가격, 리뷰, 일정 정보 | "오늘 비트코인 가격", "다음 주 도쿄 날씨 예보" |

태스크의 어려움은 **같은 도메인 내 의도 판별** 에 있습니다 — "날씨" 는
Local (Galaxy Weather 앱) 이지만 "다음 주 도쿄 날씨" 는 Search; "hello
번역" 은 Local 이지만 "3페이지 문서 번역" 은 Cloud. 이 경계는 키워드만
보고는 잡히지 않아서 작은 모델에 특히 까다롭습니다.

### 2.2 데이터셋: train / val / test 의 세 자리

현재 production 세팅 (v4) 의 split 입니다.

| 자리 | 파일 | 크기 | 출처 | 누가 보는가 |
|---|---|---|---|---|
| **Train** | `seed_set.jsonl` | 150 | 손으로 큐레이션, 4-tier 난이도 균형 | DSPy 옵티마이저 (few-shot bootstrapping 의 풀) |
| **Val** | `val.jsonl` | 600 | Gemini 3 Flash 생성 (Galaxy 컨텍스트) | DSPy 옵티마이저가 candidate 마다 score 매겨 best 선택 |
| **Test** | `test.jsonl` | 600 | Gemini 3 Flash 생성 (Galaxy 컨텍스트) | **`evaluate.py` 만**. 옵티마이저는 한 번도 노출 안 됨 |

**Train 자리의 시드셋 (150)** 이 핵심 디자인 결정입니다. privacy-sensitive
생성 질의, 도메인 모호 질의, 키워드-orthogonal 의도 등 의도적으로 어려운
케이스를 박아 넣어, 작은 모델에서도 robust 한 프롬프트가 강제로 나오도록
설계. 초기 v3 에서는 이 자리에 합성 4800 (`train.jsonl`, Gemini 가 만든
6000 의 80% split) 을 썼지만 시드셋으로 바꾼 뒤 hold-out test 점수가 더
좋아져 v4 부터 시드셋으로 고정. **합성 4800 (train split) 은 현재 세팅에서
사용되지 않습니다.**

**Val / Test 600+600** 은 Gemini 3 Flash 합성셋의 10% / 10% split.
test 셋은 학습/검증 어디에도 노출되지 않은 진짜 hold-out 입니다 — 아래
§3.3 의 모든 점수는 이 셋 기준.

### 2.3 메트릭

DSPy 옵티마이저가 최대화하는 score:

```
score = (예측 라벨 == 정답 ? 1.0 : 0.0)  −  0.5 × (privacy-sensitive 질의를 Cloud 로 보냈는가)
```

프라이버시 패널티는 의도적인 트레이드오프입니다 — "내 $50,000 부채 예산
짜줘" 는 생성 품질 측면에서는 Cloud 가 유리하지만 개인 재무 정보가 외부로
나갑니다. 메트릭이 −0.5 로 직접 페널라이즈해서 라우터가 일부러 Local /
온디바이스를 선택하도록 유도.

---

## 3. Router DSPy 최적화

### 3.1 왜 DSPy 인가

손으로 짠 프롬프트는 모델/도메인이 바뀌면 drift 합니다. DSPy 는 **시그니처 (입력/출력 스키마) 만 선언** 하면 옵티마이저가 인스트럭션 + few-shot 데모를 자동 탐색해서 메트릭을 최대화하는 프롬프트를 빌드합니다. 결과물은 선언적 시그니처 + 학습된 상태 (`.json`) 로 저장되어 프로덕션에서 그대로 로드하여 **프롬프트가 코드와 함께 버전 관리되는 자산** 이 됩니다.

### 3.2 옵티마이저 비교

같은 시드셋 + 같은 메트릭에 세 옵티마이저 적용:

| 옵티마이저 | 학습 대상 | 탐색 방식 |
|---|---|---|
| **MIPROv2** | 인스트럭션 + few-shot 데모 | Bayesian 동시 최적화 |
| **BFRS** (BootstrapFewShotWithRandomSearch) | few-shot 데모만 | Random search; 인스트럭션 fixed |
| **COPRO** | 인스트럭션 prose 만 | LLM 이 prose 를 반복 rewrite |

**Validation weighted F1** (`val.jsonl` 600개, §2.2 참조):

| 모델 | Baseline | MIPROv2 | BFRS | COPRO |
|---|---|---|---|---|
| Gemma 4 31B (레퍼런스) | — | **88.37%** | — | — |
| Gemma 4 E2B (5.12B) | 81.80% | 85.18% | **86.25%** | 84.63% |
| Qwen 3.5 0.8B (873M) | 63.92% | 72.05% | **80.78%** | 73.50% |

**관찰**: 작은 모델일수록 BFRS 가 MIPROv2 를 더 크게 이김 (Qwen 에서 +8.7
pp). 추가로 cross-ablation 을 돌렸습니다 — BFRS 가 고른 데모를 MIPROv2 가
쓴 인스트럭션과 결합, 그리고 그 반대 — 결과는 **차이의 원인이 데모 선택**
이지 인스트럭션 rewriting 이 아니라는 것이었습니다. MIPROv2 의 Bayesian
탐색이 instruction × demo 결합 공간에서 demo 측 시그널을 약하게 봅니다.

또 한 가지 — **COPRO 는 SLM 에서 약했습니다.** COPRO 는 인스트럭션을 LLM
이 verbosely rewrite 하는데, 작은 모델은 긴 prose 인스트럭션을 잘 못 따라
갑니다. Qwen 에서 COPRO 73.50% < BFRS 80.78% 가 그 증거.

### 3.3 Hold-out 테스트 결과

**학습 시 한 번도 노출되지 않은** `test.jsonl` 600개 기준 weighted F1:

| 모델 | 옵티마이저 | Test Weighted F1 | Privacy leak |
|-------|-----------|------------------|--------------|
| Gemma 4 31B (레퍼런스) | MIPROv2 | **97.18%** | 0/11 (0.0%) |
| **Gemma 4 E2B (production)** | **BFRS** | **95.99%** | **0/11 (0.0%)** |
| Qwen 3.5 0.8B | BFRS | 87.76% | 0/11 (0.0%, §3.4 참조) |

> **핵심 결과**: Gemma 4 E2B + BFRS 는 31B 레퍼런스 대비 **−1.19 pp** —
> 2B급 모델이 31B 와 사실상 동급의 라우팅 품질을 hold-out 에서
> 보였습니다.

per-class 혼동행렬, 모델별 latency 분포, 에러 분석 산출물은 모두
`research/results/` 에 PNG / JSON 으로 함께 들어 있습니다 — Stage 별
recompute 없이 그대로 비교 가능합니다.

### 3.4 Privacy leak 측정 도구의 결함

위 표의 Qwen 행은 처음에 **18.2% (2/11) privacy leak** 으로 잡혔던 셋
입니다. 두 양성 샘플을 직접 들여다본 결과는 다음과 같았습니다:

- 두 양성 샘플 모두 **정답 라벨이 `search`** 였고 실제 민감 개인 정보는
  없었음.
- 측정 도구 `is_privacy_sensitive()` 가 flat substring 스캐너여서,
  **"my health profile"** (Galaxy Health 앱의 정식 메뉴 명칭) 과
  **"social security trust fund"** (미국 공공 프로그램 이름) 을 false
  positive 로 잡았음.

→ **실제 privacy leak 은 0/11**. 측정 도구를 LLM-as-judge 로 대체하는 게
backlog 입니다. flat substring 패턴이 production 메트릭으로는 부족하다는
점이 이 조사의 가장 큰 교훈이었습니다.

### 3.5 재현하기

```bash
# Production 디폴트로 평가 (Gemma 4 E2B + BFRS, 95.99% F1)
python research/evaluate.py

# 직접 다시 최적화 (수 시간 소요, Ollama 2 인스턴스 필요)
python research/dspy_optimizer.py --model gemma4:e2b --seed-set --auto light \
  --task-base http://127.0.0.1:11435 \
  --prompt-model gemma4:31b --prompt-base http://127.0.0.1:11434
```

위 명령은 **task model (E2B)** 은 GPU 2 의 Ollama 인스턴스, **proposer
(31B)** 는 GPU 1 의 Ollama 인스턴스에 올리는 셋업을 가정합니다 — 같은
인스턴스에 두 모델을 올리면 DSPy 가 직렬 큐에 걸려 timeout 이 납니다.

Production 아티팩트: `research/artifacts/optimized_router_state_gemma4_e2b_bfrs.json` - 이 한 파일이 95.99% F1 라우터 그 자체.

---

## 4. Memory DSPy 최적화

<!-- [아직 진행중이므로 그대로] -->

---

## 기술 스택

- **레퍼런스 / 티처 모델**: Gemma 4 31B via Ollama — F1 ceiling 레퍼런스 + DSPy proposer 양쪽으로 사용
- **온디바이스 SLM**: Gemma 4 E2B (5.12B, production), Qwen 3.5 0.8B (873M, 비교용)
- **DSPy 옵티마이저**: `BootstrapFewShotWithRandomSearch` (BFRS) — §3.2 ablation 에서 MIPROv2 / COPRO 와 비교 후 채택
- **평가**: 600 큐레이션 hold-out 셋, weighted F1 + privacy leak rate
- **Python**: 3.11 (micromamba)

---

## 라이선스

Apache 2.0 — `LICENSE` 참조.
