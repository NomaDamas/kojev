# kojev - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** KoJev — a Korean "System One" decision model: give it a situation in Korean plus typed questions (pick-one, rating, yes/no) and it returns calibrated probabilities for every answer in a single pass, plus a benchmark report showing how well-calibrated and accurate it is against a held-out Korean benchmark, and a small local API to call it.

**Why this approach:** We follow the only public, measured reproduction of Jev's shape (encoder + scoring head, calibration-aware loss, question augmentation) and swap in the strongest Korean encoder built on the ModernBERT architecture; synthetic labels come from a small open Qwen teacher only where no human labels exist, under a strict $100 spend cap.

**What it will NOT do:** It never trains on the evaluation benchmark, never spends past $100 on the labeling API, and does not become a public hosted service.

**Effort:** XL
<!-- Effort is exactly ONE band, never hours/days. Quick = single edit, minutes of agent work; Short = one focused change, a few files; Medium = multi-file feature in one session; Large = several waves, one long session; XL = multi-session or architectural work. A written duration is rewritten to a band. -->
**Risk:** Medium - the chosen backbone family has documented run-to-run training instability, mitigated by multi-seed runs and a diagnostic control model.
**Decisions to sanity-check:** backbone = skt/A.X-Encoder-base (user-approved); teacher = qwen/qwen3-vl-8b-instruct (user-approved); AI Hub = optional non-blocking lane (user-approved); KoBEST fully held out as the benchmark; RLCD stage is a best-public-guess design gated so it ships only if it measurably improves calibration.

Your next move: run `/ulw-execute` on this plan (plan-reviewer high-accuracy review already required and recorded). Full execution detail follows below.

---

> TL;DR (machine): XL / Medium risk / deliverables: KoJev repo (schema+data+model+train+distill+rlcd+eval+serve), trained A.X-Encoder-base typed-decision checkpoints (3 seeds + gated RLCD), KoBEST+KLUE+OOD benchmark report, $100-capped Qwen distillation corpus, slurm pipeline on gpu01 at /data2/jeffrey/kojev.

## Scope
### Must have
- Typed-decision schema and model: state + N questions (choice <=255 options / score 2-10 ordered levels / noul) -> per-question probability distribution, expected-level score, p(yes) noul, confidence = 1 - H(p)/ln K; one forward pass; zero structured-output errors by construction.
- Backbone `skt/A.X-Encoder-base` (ModernBERT architecture, 16k ctx) + span-pooling scoring head: input `[CLS][STATE] state [Q] instr [OPT] opt1 [OPT] opt2 ... [SEP]` with 3 added marker tokens; head scores `[mean(question text tokens); mean(option text tokens); elementwise product]` -> MLP -> softmax within each question's option group. NEVER a marker-token-hidden-state head (measured not to learn: kotoba-lang/typed-decisions ablation).
- Loss CE + Brier; post-hoc temperature fitted on val; gold-preserving augmentation p=0.7 (option shuffle / instruction paraphrase templates / distractor drop / score-level synonym relabel / noul negation with flipped gold).
- Korean gold corpus from verified ungated HF datasets (exact IDs in todo 3) with per-source train/val/test split, plus a rule-gold Korean OOD question set (unseen instructions + unseen option lists + negated nouls + monotone score relabels).
- KoBEST (`skt/kobest_v1`) fully HELD OUT of training as the zero-shot benchmark; KLUE dev as secondary; open-jev-deberta-v3-large run on the same Korean benchmark as a public baseline arm.
- Qwen distillation: `qwen/qwen3-vl-8b-instruct` via OpenRouter labels GOLD-LESS question families only (never replacing gold); hard budget cap $100 enforced in code with a per-request usage ledger that halts at $95.
- RLCD-style stage 2: differentiable expected-decision-utility + soft-calibration (softECE) + KL-anchor-to-SFT objective with a GO/NO-GO gate vs SFT-only.
- >=3 SFT seeds on the main arm; `kakaobank/kf-deberta-base` single-seed diagnostic control arm (data-bug vs backbone-instability disambiguation only, not a product arm).
- All heavy work on gpu01 via Slurm (partition `batch`, `--gres=gpu:rtx6000:1`), workdir `/data2/jeffrey/kojev`; repo synced from local `/Volumes/SSD1/1/KoJev`.
- Minimal serving shim: Python `decide()` API + local FastAPI `/v1/systemone`-compatible endpoint.
- AI Hub optional lane: user-facing acquisition instructions + ingestion hook; never blocks the pipeline.
### Must NOT have (guardrails, anti-slop, scope boundaries)
- NO training or hyperparameter selection on any KoBEST split (benchmark contamination).
- NO OpenRouter spend beyond $100 total; no other paid APIs.
- NO pretraining/continued-pretraining of a backbone from scratch.
- NO public deployment, no HF Hub upload unless the user later asks.
- NO writes to `/`, `/data`, `/data1` on the cluster (disks full); everything under `/data2/jeffrey/kojev`.
- NO use of TypeSafe's API or data as training gold; no gated/licensed datasets beyond what the user personally downloads from AI Hub.
- NO teacher labels overriding dataset gold (measured to hurt in-domain accuracy).
- NO consistency-KL loss in SFT (measured: no gain, 2x cost); no `.only`/skip games in tests.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD (pytest; tiny-random-model tests run on CPU locally and on gpu01 login node; every behavioral module gets a failing-first test). Training-quality criteria are verified by agent-executed metric gates on `report.json` artifacts, not by unit tests.
- Evidence: .omo/evidence/task-<N>-kojev.<ext> (local); cluster artifacts under /data2/jeffrey/kojev/runs/<run-id>/report.json, synced back to .omo/evidence/ for the record.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.
- Wave 1 (foundation, all parallel): 1 repo+schema, 2 gpu01 workspace+slurm templates, 3 gold corpus builder, 4 benchmark harness, 5 AI Hub lane doc
- Wave 2 (modules, all parallel after their deps): 6 model module, 7 augmentation, 8 OOD set builder, 9 OpenRouter distillation client + budget ledger
- Wave 3: 10 SFT trainer + slurm smoke run, 11 teacher labeling runs
- Wave 4: 12 full SFT runs (3 seeds + control + distill arm), 13 RLCD stage + gated runs
- Wave 5: 14 evaluation + benchmark report, 15 serving shim, 16 final report + packaging
- Fan-out rule: within a wave, todos have disjoint write scopes (different modules/dirs) and MAY run as parallel subagents; cluster runs are slurm jobs watched via monitors, never polled.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 3,4,6,7,8,9,10 | 2,3,4,5 |
| 2 | - | 10,11,12,13,14 | 1,3,4,5 |
| 3 | 1 | 8,10,11,12 | 2,4,5,6,7,9 |
| 4 | 1 | 14 | 2,3,5,6,7,8,9 |
| 5 | - | (nothing; optional lane) | all |
| 6 | 1 | 10,15 | 3,4,7,8,9 |
| 7 | 1 | 10 | 3,4,6,8,9 |
| 8 | 3 | 14 | 6,7,9 |
| 9 | 1 | 11 | 6,7,8 |
| 10 | 2,3,6,7 | 12 | 11 |
| 11 | 2,3,9 | 12 (distill arm only) | 10 |
| 12 | 10,11 | 13,14,15 | - |
| 13 | 12 | 14 | - |
| 14 | 4,8,12,13 | 16 | 15 |
| 15 | 6,12 | 16 | 14 |
| 16 | 14,15 | - | - |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Repo scaffold + typed-decision schema
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: In /Volumes/SSD1/1/KoJev: `git init`; create remote: `gh repo create NomaDamas/kojev --private --source=. --remote=origin --push` (gh authed as vkehfdl1, org NomaDamas verified accessible 2026-09-19, repo does not yet exist; ssh protocol); every subsequent todo's commit is PUSHED to origin the moment it lands (`git push` at end of each todo — push failure blocks the todo's done transition); uv project (Python 3.12) named `kojev`, deps: torch, transformers>=4.51, datasets, safetensors, accelerate, fastapi, uvicorn, httpx, pytest. Package `kojev/` with `schema.py`: dataclasses `Question` (type: choice|score|noul; instructions: str; options: list[str] for choice 2..255 / score 2..10 ordered levels; gold: int|None; meta: dict) and `Example` (state: str; questions: list[Question]; source: str; split: str), JSONL (de)serialization, validation errors on malformed input (score with 1 or 11 levels rejected; choice >255 rejected). Confidence util `1 - H(p)/ln K`. Must NOT: no model code here, no network.
  Parallelization: Wave 1 | Blocked by: - | Blocks: 3,4,6,7,8,9,10
  References: Jev shape + API fields: https://docs.typesafe.ai/concepts/system-one , https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large (Use section shows decide() I/O shape); OpenJev wire shape: https://github.com/razorback16/openjev#api
  Acceptance criteria (agent-executable): `uv run pytest tests/test_schema.py -q` green; round-trip JSONL test passes; invalid inputs raise.
  QA scenarios: happy = `uv run python -c "from kojev.schema import Example,Question; ...roundtrip..."` prints parsed example; failure = score question with 1 level raises ValueError (asserted in test). Evidence .omo/evidence/task-1-kojev.txt
  Commit: Y | feat(schema): typed-decision schema + jsonl io

- [x] 2. gpu01 workspace + slurm job templates + sync script
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: `ssh gpu01 'mkdir -p /data2/jeffrey/kojev/{repo,data,runs,hf-cache}'`; `scripts/sync.sh` = rsync local repo -> gpu01:/data2/jeffrey/kojev/repo (exclude .git,.venv,data). `scripts/slurm/` templates: `smoke.sbatch` (partition interactive, --gres=gpu:rtx6000:1, --time=00:30:00, --cpus-per-task=8, --mem=32G), `train.sbatch` (partition batch, --gres=gpu:rtx6000:1, --time=24:00:00, --cpus-per-task=12, --mem=64G), both `export HF_HOME=/data2/jeffrey/kojev/hf-cache` and `cd /data2/jeffrey/kojev/repo && uv sync && uv run ...` with args passthrough. Remote env bootstrap: `uv venv` + `uv sync` on gpu01 (uv is at /usr/local/bin/uv). Must NOT: write to /, /data, /data1; no sudo; do not touch other users' dirs on /data2.
  Parallelization: Wave 1 | Blocked by: - | Blocks: 10,11,12,13,14
  References: probed 2026-09-19: node gpu01 Gres=gpu:rtx6000:3, 64 CPU, 380G RAM; partitions batch*(7d)/interactive(12h); Python 3.12.3; internet OK; /data2 9.9T free writable (per-user dir convention). ssh alias `gpu01` works BatchMode.
  Acceptance criteria: `ssh gpu01 'cd /data2/jeffrey/kojev/repo && uv run python -c "import torch; print(torch.cuda.is_available())"'` prints True via `srun -p interactive --gres=gpu:rtx6000:1`; sbatch smoke job reaches COMPLETED in `sacct`.
  QA scenarios: happy = `./scripts/sync.sh && ssh gpu01 sbatch /data2/jeffrey/kojev/repo/scripts/slurm/smoke.sbatch` then `sacct -j <id> --format=State` shows COMPLETED; failure = template with wrong gres name rejected by sbatch (capture error once, then fix). Evidence .omo/evidence/task-2-kojev.txt
  Commit: Y | feat(infra): gpu01 sync + slurm templates

- [x] 3. Gold corpus builder (Korean HF datasets -> typed decisions + splits)
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/data_gold.py`. Sources (ALL verified ungated on HF 2026-09-19) and question mappings (gold from each dataset's own label; deterministic rules only):
  - `e9t/nsmc` (config default, fields document/label): choice polarity 2 (부정/긍정) + noul "이 리뷰는 긍정적이다" (label) + negated-form noul at build time only for OOD, not train.
  - `klue/klue` config `ynat` (title/label 7): choice topic 7 (IT과학/경제/사회/생활문화/세계/스포츠/정치) + noul "정치 기사이다".
  - `klue/klue` config `nli` (premise+hypothesis/label 3): state="전제: {premise}\n가설: {hypothesis}", choice 3 (함의/중립/모순) + noul "가설은 전제에 의해 함의된다".
  - `klue/klue` config `sts` (sentence pair, labels.real-label 0..5): score 6 ordered levels (전혀 다름..완전히 같음) gold=round(real-label); noul "두 문장은 사실상 같은 의미다" (binarized at 3.0, klue sts binary-label field).
  - `kakaobrain/kor_nli` (mnli+snli train subsets): choice 3 as above (cap 8k states).
  - `smilegate-ai/kor_unsmile` (문장 + 10 binary cols): nouls per category (여성/가족 혐오, 남성 혐오, 성소수자 혐오, 인종/국적 혐오, 연령 혐오, 지역 혐오, 종교 혐오, 기타 혐오, 악플/욕설, clean) 3 sampled per state + choice "주된 혐오 유형" over argmax when exactly one category is 1.
  - `jeanlee/kmhas_korean_hate_speech` (multi-label 9): 3 sampled nouls per state.
  - `searle-j/kote` (44 emotion multi-label): nouls over 8 curated frequent emotions + choice 4 coarse (기쁨/슬픔/분노/중립) via fixed emotion->coarse rule table written in code.
  - `kor_3i4k` (7 intent classes): choice 7 + noul "질문이다".
  - `lawcompany/KLAID` (ljp fact/laws): choice over the 177 statute classes (<=255 OK) capped 6k states + noul "형법 조항이 적용된다" (rule over statute name prefix).
  - `KorQuAD/squad_kor_v1` (context/question): noul "이 지문으로 질문에 답할 수 있다" — positive = own question, negative = question sampled from a different article (seed 0); 8k states.
  Per-dataset primitive matrix (target train questions; states capped 8000/source):
  | dataset | state | choice | score | noul | ~train q |
  | --- | --- | --- | --- | --- | --- |
  | e9t/nsmc | review | polarity 2 | - | 긍정이다 | 16k |
  | klue/ynat | headline | topic 7 | - | 정치기사이다 | 16k |
  | klue/nli | premise+hypothesis | 함의/중립/모순 3 | - | 함의된다 | 16k |
  | klue/sts | sentence pair | - | 유사도 6단계 | 사실상 같은 의미다 | 16k |
  | kakaobrain/kor_nli | premise+hypothesis | 3 | - | - | 8k |
  | smilegate-ai/kor_unsmile | comment | 주 혐오유형 (단일라벨 행만) | - | 카테고리별 3개 | 24k |
  | jeanlee/kmhas_korean_hate_speech | comment | - | - | 카테고리별 3개 | 24k |
  | searle-j/kote | sentence | coarse 감정 4 | - | 큐레이션 8감정 중 1개 | 16k |
  | kor_3i4k | sentence | 의도 7 | - | 질문이다 | 16k |
  | lawcompany/KLAID | 사실관계 | 적용 법조 177 | - | 형법 조항이다 | 12k |
  | KorQuAD/squad_kor_v1 | 지문+질문 | - | - | 답할 수 있다 | 8k |
  | **합계** | | | | | **~172k (acceptance floor 90k)** |
  Split: per-source shuffled seed 0, cap per source train<=8000 / val<=500 / test<=1000 STATES; test set shuffled across sources. Emit data/gold/{train,val,test}.jsonl + counts table in data/gold/summary.json. Must NOT: touch skt/kobest_v1 here; no AI Hub; no teacher labels; KorQuAD dev article ids reserved for OOD builder (todo 8) - exclude from train/val.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 8,10,11,12
  References: mapping style mirrors kotoba-lang/typed-decisions "What the labels are" section; verified IDs in .omo/drafts/kojev.md Findings; kakaobrain/kor_sts is 401-gated -> NOT used (KLUE sts covers it).
  Acceptance criteria: `uv run python -m kojev.data_gold --out data/gold` completes; summary.json shows >=9 sources, >=40k train states total, >=90k train questions; `uv run pytest tests/test_data_gold.py -q` green (per-source mapper unit tests on fixture rows incl. label-flip checks for negations and the KorQuAD negative-sampling rule).
  QA scenarios: happy = build then `uv run python -m kojev.data_gold --validate data/gold` re-parses every line through schema validation with 0 errors; failure = corrupt one line and --validate exits nonzero naming the line. Evidence .omo/evidence/task-3-kojev.txt
  Commit: Y | feat(data): korean gold corpus builder + splits

- [x] 4. Benchmark harness: KoBEST + KLUE dev, zero-shot protocol
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/bench.py`. Map `skt/kobest_v1` configs to typed questions: boolq (passage+question -> noul), copa (premise + 2 alternatives -> choice 2, 원인/결과 instruction from the question field), wic (word + 2 contexts -> noul "같은 의미로 쓰였다"), hellaswag (context -> choice 4 endings), sentineg (sentence -> noul 긍정). KLUE dev: ynat/nli/sts mapped as in todo 3 but from validation split. Output per-task accuracy + macro-F1 + Brier + ECE (15-bin) + per-question-kind aggregates into a single benchmark-report.json; runner takes any model implementing `decide(state, questions)`. Must NOT: benchmark data must never be written into data/gold; harness asserts at runtime that no example id from kobest appears in the training manifest.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 14
  References: KoBEST fields: https://huggingface.co/datasets/skt/kobest_v1 ; A.X-Encoder card reports fine-tuned KoBEST f1 (boolq 84.5/copa 78.7/sentineg 96.0/wic 80.8) — context for (lower) zero-shot expectations, not targets.
  Acceptance criteria: `uv run pytest tests/test_bench.py -q` green (mapping unit tests on fixture rows; a uniform-random model gets ~chance accuracy and ECE>0.1 on a 200-example smoke slice); `uv run python -m kojev.bench --model random --limit 200` writes benchmark-report.json.
  QA scenarios: happy = random-model smoke run report has all 5 kobest tasks + 3 klue tasks with counts>0; failure = passing a training-manifest containing a kobest id makes the contamination assert fire. Evidence .omo/evidence/task-4-kojev.txt
  Commit: Y | feat(eval): kobest+klue zero-shot benchmark harness

- [x] 5. AI Hub optional lane: user instructions + ingestion hook
  Recommended task executor category: writing
  What to do / Must NOT do: `docs/aihub.md` in Korean for the user: (1) aihub.or.kr 회원가입 + 본인인증, (2) 데이터셋 페이지에서 활용신청(목적 기재, 대부분 자동승인, 일부 1-3일 심사), (3) 마이페이지에서 API key 발급, (4) gpu01에서 aihubshell 다운로드(`curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do; chmod +x aihubshell; ./aihubshell -mode d -datasetkey <데이터셋번호> -aihubapikey <API키>`) to /data2/jeffrey/kojev/data/aihub, (5) 라이선스: 연구목적 외 재배포 금지 명시. Recommend 2 datasets with typed-decision mappings: 감성 대화 말뭉치(감정 score/choice), 민원(콜센터) 질의응답(의도 choice + 긴급도 score). Ingestion hook: `kojev/data_aihub.py` stub that maps a downloaded dir to Example JSONL, raising NotImplementedError with the mapping doc when dir absent — pipeline runs fine without it. Must NOT: no scraping, no credentials in repo, nothing in the default pipeline depends on this lane.
  Parallelization: Wave 1 | Blocked by: - | Blocks: -
  References: aihubshell usage from aihub.or.kr 개발자 가이드 (verify exact URL at execution; if CLI unavailable, document browser download + scp path instead).
  Acceptance criteria: docs/aihub.md exists covering steps 1-5 + 2 mappings; `uv run python -c "import kojev.data_aihub"` imports clean; `uv run pytest tests/test_data_aihub.py -q` green (absent-dir raises with doc pointer).
  QA scenarios: happy = doc renders (read back, all 5 steps present); failure = data_aihub on missing dir raises NotImplementedError mentioning docs/aihub.md. Evidence .omo/evidence/task-5-kojev.txt
  Commit: Y | docs(aihub): optional acquisition lane + ingestion stub

- [x] 6. Model module: A.X-Encoder + span-pooling grouped-softmax head
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/encoder.py`. Tokenizer: add [STATE],[Q],[OPT] special tokens to `skt/A.X-Encoder-base` tokenizer, resize embeddings. Collator packs state + all questions of an example into one sequence (truncate state first at budget, ctx cap 4096 for training); per-option index maps for span pooling. Head: for each option, features [mean(instr token hiddens); mean(option token hiddens); product] -> MLP (3-layer, GELU, hidden 1024) -> scalar logit; softmax within question group. Loss = CE + 1.0*Brier over the group. score readout = expected level index; noul = p(yes). `KoJevModel.decide()` for inference incl. batching. Must NOT: no marker-token-hidden readout; no generation; fp32 master + bf16 autocast.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 10,15
  References: head/pooling ablation and formula: https://github.com/kotoba-lang/typed-decisions ("The ablation that mattered", "How it works" on the HF card); A.X quickstart transformers>=4.51.
  Acceptance criteria: `uv run pytest tests/test_encoder.py -q` green using a tiny-random ModernBERT config (no download): shapes correct, grouped softmax sums to 1 per question, 16-example overfit reaches loss <0.01 in <=200 steps on CPU (the recipe's mechanics check), decide() returns schema-valid answers for mixed choice/score/noul.
  QA scenarios: happy = overfit test output pasted; failure = feeding a 300-option choice raises schema validation before the model. Evidence .omo/evidence/task-6-kojev.txt
  Commit: Y | feat(model): span-pooling grouped-softmax head on A.X encoder

- [x] 7. Gold-preserving augmentation module
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/augment.py` applied at train time with prob p (default 0.7): (a) choice option-order shuffle (gold index remapped); (b) instruction paraphrase via fixed Korean template bank >=8 per question family (written in code, e.g. "~은/는 무엇인가?" / "다음 중 ~을 고르시오" variants with correct 은/는·이/가 josa handling by final-consonant rule); (c) distractor drop for choice K>=4 (never drops gold); (d) score level-name synonym swap preserving order (매우 부정→아주 나쁨 etc.); (e) noul negation with flipped gold ("~이다"->"~이 아니다", gold 1-g). Must NOT: any transform that can silently break gold (each transform unit-tested for gold preservation); no consistency-KL.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 10
  References: measured effects (OOD +2.6pt, negation noul +16pt, OOD ECE -0.03 at p=0.7; consistency-KL no gain): typed-decisions 第3反復 table.
  Acceptance criteria: `uv run pytest tests/test_augment.py -q` green: 1000 random examples x all transforms -> re-evaluating gold under the transformed surface matches the remapped gold 1000/1000; negation flips exactly; josa correctness spot tests.
  QA scenarios: happy = property test log; failure = a transform returning dropped-gold option set is caught by the invariant test. Evidence .omo/evidence/task-7-kojev.txt
  Commit: Y | feat(augment): gold-preserving korean question augmentation

- [x] 8. Korean OOD question set (rule gold, unseen surfaces)
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/data_ood.py` builds data/ood/test.jsonl over GOLD TEST states only (never train/val): unseen instructions + unseen option vocab per source, gold via deterministic rules: (a) nsmc: noul 부정형 "이 리뷰는 긍정적이지 않다" (flipped gold) + score 2 monotone levels (불만족/만족, gold = polarity) and its REVERSED-direction variant (만족/불만족, gold flipped) — the monotone-reversal probe on a binary source; (b) ynat: choice over 4 coarse merged topics (rule table over 7 labels) + noul "경제 관련 기사다"; (c) nli: noul "전제와 가설은 모순된다" + reversed-direction instruction; (d) sts: score with REVERSED direction levels (완전히 같음..전혀 다름, gold = 5-orig) — the monotone-reversal probe; (e) korquad: noul "이 지문은 질문과 무관하다" (flipped); (f) unsmile: choice "혐오 없음/혐오 있음" 2-way from clean col. Every OOD instruction string must NOT appear in train (asserted). Emit majority-class baseline per question in summary. Must NOT: no teacher labels here; drop any question whose rule gold is debatable (the recipe deleted 2 such probes — follow that bar).
  Parallelization: Wave 2 | Blocked by: 3 | Blocks: 14
  References: OOD methodology + the deleted-probe lesson: typed-decisions 第2/第4反復 OOD sections.
  Acceptance criteria: `uv run python -m kojev.data_ood --gold data/gold --out data/ood` emits >=2500 questions across >=6 families; `uv run pytest tests/test_data_ood.py -q` green (flip rules, reversal rule, train-overlap assert).
  QA scenarios: happy = summary.json lists majority baseline per family; failure = injecting a train instruction into OOD set trips the overlap assert. Evidence .omo/evidence/task-8-kojev.txt
  Commit: Y | feat(data): rule-gold korean OOD question set

- [x] 9. OpenRouter distillation client + $100 hard budget ledger
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/teacher.py`: async client for `qwen/qwen3-vl-8b-instruct` (OPENROUTER_API_KEY from env — present on gpu01 and local), HIGH PARALLELISM: adaptive concurrency starting 32, ramping to 64 in-flight, client-side token-bucket + exponential backoff on 429/5xx (halve concurrency on 3 consecutive errors, recover 2x per 60s quiet window), line-oriented answer format ("A1: <option index>" per question) + tolerant parser (the recipe's parser lost 19%->4% with line-oriented format — start there), request `usage` accounting: every response's prompt/completion tokens * ($0.117/$0.455 per M) appended to data/distill/ledger.jsonl; client REFUSES new requests when cumulative cost >= $95 (buffer to $100) and exits 0 with a BUDGET_REACHED marker. Try `logprobs` once; if provider returns them, store top-5 as soft labels, else hard labels. Prompt template (Korean) pins: 질문당 한 줄, 선택지 번호만. Must NOT: never call any model other than the pinned id; never exceed ledger cap even across restarts (ledger is the source of truth, loaded at start); no teacher calls on gold-labeled families' training questions.
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 11
  References: pricing verified 2026-09-19 via openrouter.ai/api/v1/models (qwen/qwen3-vl-8b-instruct ctx 262k, $0.117/M in $0.455/M out); teacher-economics lessons (hard labels, parser, concurrency hangs): typed-decisions 第2反復 ② and 第4反復 4.
  Acceptance criteria: `uv run pytest tests/test_teacher.py -q` green with a mocked HTTP layer: cost accounting exact to the token, cap halts at $95 across simulated restart, parser handles the 3 observed answer-format deviations; one REAL 3-request probe against OpenRouter completes and ledger shows cost < $0.01.
  QA scenarios: happy = real probe output + ledger lines; failure = mock returning cost pushing total to $95.01 -> client refuses further requests and writes BUDGET_REACHED. Evidence .omo/evidence/task-9-kojev.txt
  Commit: Y | feat(distill): budget-capped openrouter teacher client

- [x] 10. SFT trainer + report.json + slurm smoke run
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/train.py`: dataloader over gold JSONL with augmentation p, AdamW (backbone lr 2e-5, head lr 1e-3, warmup 6%, cosine), CE+Brier, bf16 autocast, grad clip 1.0, DIVERGENCE WATCH (if running-mean train loss rises >25% over 200 steps or is NaN: abort run, mark report diverged=true — ModernBERT-family documented instability), eval each 0.25 epoch on val (acc/Brier/ECE per kind+source), post-hoc temperature fit on val, everything into runs/<run-id>/report.json (args, data counts, loss curve, wall, peak mem, metrics, temperature). Then: sync to gpu01, `sbatch scripts/slurm/smoke.sbatch -- python -m kojev.train --limit 2000 --epochs 1 --seed 0` on interactive partition; verify report.json lands and val accuracy beats majority baseline by >=5pt on the 2k slice. Must NOT: no training on kobest; no >4096 ctx in this stage; smoke uses --limit, never the full corpus on interactive.
  Parallelization: Wave 3 | Blocked by: 2,3,6,7 | Blocks: 12
  References: hyperparam neighborhoods from typed-decisions measured rows (DeBERTa lr 2e-5 1ep best; base-model 5e-5; 2ep memorizes); divergence signatures (loss 1.5->6.1).
  Acceptance criteria: smoke slurm job COMPLETED (`sacct`); runs/<id>/report.json parses, has all required keys, val_acc_all - majority_all >= 0.05; `uv run pytest tests/test_train.py -q` green (loss computation, divergence-watch triggers on synthetic NaN batch, temperature-fit unit test).
  QA scenarios: happy = sacct line + report.json key dump; failure = synthetic diverging loss stream trips the watch and report has diverged=true (unit test). Evidence .omo/evidence/task-10-kojev.txt
  Commit: Y | feat(train): sft loop + calibration + divergence watch + slurm smoke

- [x] 11. Teacher labeling runs (gold-less Korean families)
  Recommended task executor category: deep
  What to do / Must NOT do: Build gold-less question families over fresh Korean states: (a) KorQuAD train contexts (excluded article ids ok here — they are train-side): choice "이 지문의 주제" over 8 fixed topics, noul "이 지문은 수치/통계를 포함한다", score "지문의 난이도" 3 levels; (b) `daekeun-ml/naver-news-summarization-ko` articles: choice 기사 유형 6, noul "기업 보도자료다", score "사실적↔의견적" 3 levels; (c) OOD-STYLE families over gold train states (reading-comprehension style questions the gold sets lack). Target 120k teacher-labeled questions (~40k states x 3 q): budget math at ~1100 in + 60 out tokens/state-call => ~44M in + 2.4M out ~= $6.2; with 2x retry margin and probes <= $15 total — far under cap; cap still enforced by ledger. Throughput target at concurrency 32-64: 40k calls in 1-2h (user-directed parallelism; if provider throttles below 5 req/s sustained, accept slower). Run on gpu01 login node (network job, no GPU) inside tmux-free `nohup` slurm-less script with resume-from-ledger. Output data/distill/train.jsonl with `label_source: "teacher:qwen3-vl-8b-instruct"` per question — NEVER written into data/gold. Must NOT: no teacher labels for families that have dataset gold; stop at BUDGET_REACHED cleanly.
  Parallelization: Wave 3 | Blocked by: 2,3,9 | Blocks: 12
  References: teacher value is gold-less families only (in-domain -15pt, OOD +5pt): typed-decisions 第2反復 ② + 見積りの訂正.
  Acceptance criteria: data/distill/train.jsonl >= 100k questions with valid schema; ledger total <= $20; resume test: kill the run mid-flight, restart, no duplicate request ids, total count consistent.
  QA scenarios: happy = summary counts + ledger tail; failure = restart-after-kill produces no dupes (asserted by request-id set size == line count). Evidence .omo/evidence/task-11-kojev.txt
  Commit: Y | feat(distill): teacher-labeled goldless korean families

- [x] 12. Full SFT runs: 3 seeds + control arm + distill arm
  Recommended task executor category: deep
  What to do / Must NOT do: sbatch on batch partition, 5 runs: (A) A.X-Encoder-base, gold only, augment 0.7, 1 epoch, seeds 0/1/2; (B) kakaobank/kf-deberta-base seed 0 gold-only (diagnostic control — if A collapses but B trains, it's backbone instability, not data); (C) A.X seed 0, gold + distill families mixed (distill questions weighted 0.5 in loss). Pick main checkpoint = median val-acc seed of (A); record seed spread. If >=2 of 3 (A) runs diverge: retry once at lr 1e-5 + warmup 10%; if still diverging, ESCALATE to user with (B) evidence rather than silently switching backbone. Must NOT: no cherry-picking best seed for the report (median rule); no kobest anywhere.
  Parallelization: Wave 4 | Blocked by: 10,11 | Blocks: 13,14,15
  References: seed protocol + spread expectations (in-domain ±1pt, OOD ±2.5pt): typed-decisions 第4反復 2; instability mitigations from this plan's Decisions.
  Acceptance criteria: 5 report.json files COMPLETED, diverged=false on >=2 of 3 (A) seeds; main arm val acc-all >= 0.75 AND >= control arm (B) - 3pt; distill arm (C) val acc on gold families within 1pt of (A) median (teacher data must not hurt gold performance).
  QA scenarios: happy = table of 5 runs (seed, acc, Brier, ECE, diverged) generated by `uv run python -m kojev.report --runs runs/`; failure = a diverged run is visibly flagged and excluded by the median rule (exercise with the diverged smoke artifact if none diverges naturally). Evidence .omo/evidence/task-12-kojev.txt
  Commit: Y | feat(train): full sft arms (3 seeds + control + distill)

- [x] 13. RLCD-style stage 2 with GO/NO-GO gate
  Recommended task executor category: deep
  What to do / Must NOT do: `kojev/rlcd.py`, initialized from todo-12 main checkpoint. Because the model outputs an explicit distribution, optimize EXPECTED DECISION UTILITY directly (differentiable, no sampling): per question, soft-selection s = sigmoid((max_prob - tau)/kappa) models the act/escalate gate; L_util = -[ s * (p_gold * U_ok + (1-p_gold) * U_err) + (1-s) * U_esc ] with U_ok=+1, U_err=-4, U_esc=0, tau=0.6, kappa=0.05. L = L_util + lambda_cal * softECE(15 soft bins) + beta * KL(p || p_sft_frozen); defaults lambda_cal=0.5, beta=0.1, lr 1e-5, max 0.5 epoch, eval every 200 steps, early stop on val Brier rising 2 consecutive evals. Data: gold train + teacher-labeled families (teacher label plays outcome for L_util on gold-less data). GO/NO-GO GATE (val, vs SFT main): KEEP only if [ECE improves >= 0.005 OR selective accuracy@confidence>=0.6 improves >= 2pt] AND [in-domain acc drop <= 0.5pt] AND [Brier drop <= 0.005] AND [OOD acc drop <= 1pt]. If NO-GO after 2 hyperparameter attempts (beta 0.1->0.3, lambda_cal 0.5->1.0): ship SFT-only, record the negative result in the final report — that is a valid outcome, not a failure. Must NOT: no REINFORCE-on-gold (redundant with CE in expectation — documented in module docstring); no gate metric ever computed on test or kobest.
  Parallelization: Wave 4 | Blocked by: 12 | Blocks: 14
  References: TypeSafe's public RLCD description (one sentence: probabilities optimized against outcomes — docs.typesafe.ai/introduction/machine-learning-primer); soft calibration objectives literature (Karandikar et al. 2021) for softECE; KL-anchor stability standard from RLHF practice. This stage is a best-public-guess: no TypeSafe internals are public — stated as such in the report.
  Acceptance criteria: `uv run pytest tests/test_rlcd.py -q` green (loss differentiable end-to-end on tiny model; gate logic unit-tested on synthetic metric tables incl. all four gate clauses); one slurm RLCD run COMPLETED with gate verdict (GO or NO-GO) written into runs/<id>/report.json.
  QA scenarios: happy = gate verdict line + metric deltas table; failure = synthetic metrics violating the acc-drop clause produce NO-GO (unit test). Evidence .omo/evidence/task-13-kojev.txt
  Commit: Y | feat(rlcd): expected-utility calibration stage + go/no-go gate

- [ ] 14. Full evaluation: benchmark + OOD + latency
  Recommended task executor category: deep
  What to do / Must NOT do: Run todo-4 harness zero-shot on: main SFT checkpoint, RLCD checkpoint (if GO), control arm, distill arm, and PUBLIC BASELINE `com-kotobalabs/open-jev-deberta-v3-large` (load via its bundled loader; it is English-trained — expected weak on Korean, that IS the point of KoJev). Also run data/ood/test.jsonl. Latency on gpu01 RTX6000 via srun: 1 state x 10 questions e2e p50/p95 (20 repeats, 3 warmups) + forward-only + throughput batch 8/32. Emit eval/RESULTS.md with tables: per-task acc/F1, Brier, ECE (pre/post temperature), per-kind aggregates, majority baselines, latency rows. Must NOT: no fine-tuning on benchmark; no cherry-picked slices; every number traceable to a report.json path.
  Parallelization: Wave 5 | Blocked by: 4,8,12,13 | Blocks: 16
  References: latency protocol mirrors typed-decisions Measured section (p50/p95, warmups, forward-vs-e2e split).
  Acceptance criteria: eval/RESULTS.md exists with >=5 model rows x >=8 task columns + latency table; KoJev main beats the English baseline on >=4 of 5 KoBEST tasks; contamination assert ran clean (logged).
  QA scenarios: happy = RESULTS.md rendered + the assert log line; failure = harness on a checkpoint path that does not exist exits nonzero with a clear error (exercised once). Evidence .omo/evidence/task-14-kojev.txt
  Commit: Y | feat(eval): full benchmark + ood + latency results

- [x] 15. Serving shim: decide() + /v1/systemone endpoint
  Recommended task executor category: unspecified-low
  What to do / Must NOT do: `kojev/serve.py`: FastAPI app exposing POST /v1/systemone accepting {model, state, questions:{name:{type,instructions,criteria|options}}} (Jev wire shape) -> {model, answers:{name:{choice|score|noul, probabilities, confidence}}, usage:{input_tokens}}; maps criteria dict/list to schema options; loads the main checkpoint (env KOJEV_CKPT). 422 with {detail:{error_type,message}} on malformed questions. Local only (uvicorn 127.0.0.1). Must NOT: no auth layer, no public bind, no text generation endpoint.
  Parallelization: Wave 5 | Blocked by: 6,12 | Blocks: 16
  References: wire shape: https://github.com/razorback16/openjev#api and https://github.com/ekzhang/openjev-sglang#request .
  Acceptance criteria: `uv run pytest tests/test_serve.py -q` green (TestClient: happy request with 3 question kinds -> valid typed answers; 300-option choice -> 422; missing type -> 422).
  QA scenarios: happy = real `curl -i -X POST 127.0.0.1:8930/v1/systemone -d @tests/fixtures/req.json` against uvicorn with the trained checkpoint returns 200 + calibrated answers (capture body); then kill server pid and verify `kill -0` fails (cleanup receipt). failure = malformed body curl returns 422 with error_type. Evidence .omo/evidence/task-15-kojev.txt
  Commit: Y | feat(serve): jev-compatible local systemone endpoint

- [x] 16. Final report + packaging + budget/receipt audit
  Recommended task executor category: writing
  What to do / Must NOT do: `README.md` (repo): what KoJev is, recipe provenance (kotoba-lang/typed-decisions; TypeSafe Jev as reference point, independent, no affiliation), full results tables from eval/RESULTS.md, seed spreads, RLCD verdict (GO or negative result stated plainly), budget audit (ledger total vs $100), reproduction commands (sync -> sbatch chain), data licenses table (each HF dataset's license short row + AI Hub note), limitations (OOD ceiling expectations from the recipe, ModernBERT instability record, teacher label caveats). Package main checkpoint as local bundle /data2/jeffrey/kojev/release/kojev-v0/ (backbone HF format + head.safetensors + tokenizer + kojev_config.json with pooling/temperature/provenance) + loader round-trip test from a clean dir. Must NOT: no HF upload; no claims not backed by a report.json path; no "beats Jev" claims (not comparable — different labels).
  Parallelization: Wave 5 | Blocked by: 14,15 | Blocks: -
  References: bundle shape mirrors com-kotobalabs/open-jev-deberta-v3-large card (head.safetensors + config + loader).
  Acceptance criteria: README complete (all listed sections present); round-trip test: fresh venv on gpu01 loads release bundle and decide() returns valid answers on 3 fixture states; ledger total < $100 quoted in README matches data/distill/ledger.jsonl sum.
  QA scenarios: happy = round-trip stdout + README section checklist; failure = tampering kojev_config.json temperature to a string makes loader raise a typed error (test). Evidence .omo/evidence/task-16-kojev.txt
  Commit: Y | docs(release): final report + local bundle + budget audit

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  Recommended task executor category: unspecified-high
  Verify every todo 1-16 acceptance criterion against its evidence file; verify Must-NOT list (grep repo for kobest in data_gold paths, ledger cap logic, no /data writes in scripts); verdict APPROVE/REJECT with cited paths.
- [ ] F2. Code quality review
  Recommended task executor category: unspecified-high
  Review kojev/ modules for correctness of grouped softmax, augmentation gold-preservation, budget accounting arithmetic, RLCD gate logic; diagnostics clean; tests non-tautological (each can fail for its named regression).
- [ ] F3. Real manual QA
  Recommended task executor category: unspecified-high
  On gpu01: run one fresh `kojev.bench` invocation on the released bundle and one real curl against the serving shim (then kill it, receipt recorded); confirm numbers match eval/RESULTS.md rows.
- [ ] F4. Scope fidelity
  Recommended task executor category: unspecified-high
  Confirm nothing beyond scope shipped (no HF upload, no public bind, no extra backbones) and nothing in scope was silently dropped (AI Hub doc exists, RLCD verdict recorded either way, 3 seeds present).

## Commit strategy
- Repo starts empty: todo 1 runs `git init`; convention = Conventional Commits (`<type>(<scope>): <imperative>`, lowercase, English) — no history to mimic.
- One atomic commit per todo (its Commit line), each green on its own tests and PUSHED to `github.com/NomaDamas/kojev` (private) immediately; no WIP on main; final commit footer `Plan: .omo/plans/kojev.md`.
- Remote setup lives in todo 1 (`gh repo create NomaDamas/kojev --private`); if `gh` is unavailable at execution time, fall back to creating via GitHub web UI then `git remote add origin git@github.com:NomaDamas/kojev.git`.
- Cluster runs/ and data/ are gitignored; only code, docs, configs, and small summaries committed.

## Success criteria
1. Schema + model produce zero structured-output errors by construction: `pytest` suites of todos 1,6,15 green (evidence files 1/6/15).
2. Gold corpus >=40k train states / >=90k train questions across >=9 verified Korean sources with clean per-source train/val/test split and KoBEST absent (contamination assert log, todo 3+4 evidence).
3. Main A.X arm: >=2/3 seeds converge; median val acc-all >= 0.75, ECE (post-T) <= 0.05; control-arm comparison recorded (todo 12 evidence).
4. Distillation: >=100k teacher-labeled questions with total OpenRouter spend <= $100 enforced by ledger (todo 9/11 evidence; README budget audit todo 16).
5. RLCD stage: gate verdict recorded (GO with the mandated metric deltas, or NO-GO negative result shipped in report) — either outcome with evidence (todo 13).
6. Benchmark: zero-shot KoBEST + KLUE dev + OOD results table with Brier/ECE/latency, KoJev main beats English open-jev baseline on >=4/5 KoBEST tasks (todo 14 evidence).
7. Serving shim answers Jev-wire requests locally, cleanup receipts recorded (todo 15 evidence).
8. F1-F4 all APPROVE.

## Handoff prompt (paste into a new session to start execution)
```
ulw-execute .omo/plans/kojev.md

Context the executor needs beyond the plan (all verified 2026-09-19):
- This session planned KoJev with ulw-plan; the plan file is complete and plan-reviewer approved (draft: .omo/drafts/kojev.md, status: approved).
- Working dir /Volumes/SSD1/1/KoJev is NOT yet a git repo; todo 1 runs `git init` + `gh repo create NomaDamas/kojev --private --source=. --remote=origin --push` (gh authed as vkehfdl1; org NomaDamas confirmed). Push every commit as it lands.
- Cluster: `ssh gpu01` works BatchMode. Workdir /data2/jeffrey/kojev (already created? verify and mkdir -p subdirs). NEVER write to /, /data, /data1 (full). Slurm: partition batch (7d) / interactive (12h), node gpu01 Gres=gpu:rtx6000:3. Python 3.12.3, uv at /usr/local/bin/uv, internet OK. Set HF_HOME=/data2/jeffrey/kojev/hf-cache.
- OPENROUTER_API_KEY is in env both locally and on gpu01 — never hardcode it.
- User decisions: backbone skt/A.X-Encoder-base; teacher qwen/qwen3-vl-8b-instruct; AI Hub optional lane; OpenRouter hard cap $100 (ledger stops at $95); distillation runs with high concurrency (32→64 adaptive).
- Long-running remote jobs: submit via sbatch and WATCH with tool.monitor (never poll loops); teacher labeling runs on the login node via nohup with resume-from-ledger.
- If >=2/3 A.X SFT seeds diverge per the plan's escalation rule, stop and ask the user with the kf-deberta control-arm evidence.
```
