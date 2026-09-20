# KoJev

Korean typed-decision (System One) model: a state plus choice / score / noul
questions in, calibrated probabilities out. Local-only. No public deployment
and no Hugging Face upload.

The recipe is taken from [kotoba-lang/typed-decisions](https://github.com/kotoba-lang/typed-decisions).
[TypeSafe Jev](https://github.com/razorback16/openjev) is the wire-shape
reference (`POST /v1/systemone`). KoJev is an independent reimplementation
with no affiliation to TypeSafe, Kotoba, or OpenJev.

Backbone: `skt/A.X-Encoder-base`. Teacher: `qwen/qwen3-vl-8b-instruct` via
OpenRouter. KoBEST is fully held out of training.

## Status

Todos 1–11, 13, 15 are done. Todos 12, 14, 16 and F1–F4 are open.

- Todo 12: five `report.json` files. Arm A 3/3 healthy (0.7145 / 0.7215 /
  0.7211). Median seed 2 is **0.7211**, which **misses the 0.75 gate**.
  Distill 0.7327 does not hurt gold. Control DeBERTa 0.9132 is the 512-window
  subset, not the same val.
- Todo 13: two RLCD attempts NO-GO. Ship SFT-only.
- Todo 14: six-model `eval/RESULTS.md` (3 A.X seeds + distill + DeBERTa
  control + OpenJev). Latency protocol on gpu01. KoJev main **loses 0/5
  KoBEST** to English OpenJev (in-domain gold-val still 0.719 vs 0.522).
- Todo 16: bundle at `/data2/jeffrey/kojev/release/kojev-v0`; fresh venv
  round-trip and tampered-config EncodingError are recorded. Ledger $2.945571.

## Results

Full tables live in [eval/RESULTS.md](eval/RESULTS.md). Headline from the
median A.X gold-only seed (`sft-full-seed2`, job 13687, 103,952 train
states, 1 epoch, `diverged=false`):

| slice | acc | majority | gap |
|---|---:|---:|---:|
| overall | 0.7211 | 0.6446 | +0.0764 |
| choice | 0.4601 | 0.2417 | +0.2184 |
| noul | 0.8106 | 0.7853 | +0.0252 |
| score | 0.3900 | 0.3460 | +0.0440 |

A-seed overall: 0.7145 / 0.7215 / 0.7211 (all healthy). Distill arm 0.7327.
Control `kf-deberta-base` 0.9132 on the 512-token subset. KoBEST/KLUE/OOD/
latency tables land with todo 14. Median rule: `uv run python -m kojev.report
--runs …` (diverged rows flagged, excluded from the median).

## RLCD verdict

**NO-GO.** Two cluster attempts from median seed 2, both `stopped_early`
at 800 steps, both failed `quality` and `brier_drop`.

| attempt | job | hparams | val acc | val brier |
|---|---|---|---:|---:|
| 1 | 13703 | β=0.1 λ=0.5 gold-only | 0.7211 → 0.7201 | 0.332 → 0.361 |
| 2 | 13704 | β=0.3 λ=1.0 gold+distill | 0.7211 → 0.7189 | 0.332 → 0.347 |

Ship SFT-only. Negative result, not a training crash. Objective is expected
utility + softECE + KL to frozen SFT — not REINFORCE-on-gold.

## Budget audit

OpenRouter ledger: `data/distill/ledger.jsonl`.

```text
$ uv run python -m kojev.release audit data/distill/ledger.jsonl
{"total_usd": 2.9455706409999483, "records": 37557, "under_cap": true}
```

Cap is $100. New work stops at $95. Teacher labeling (todo 11) produced
33,340 states / 100,020 questions with unique request IDs and no duplicate
paid calls on resume. This README's `$2.945571` matches the ledger sum.

## Reproduction

Cluster writes stay under `/data2/jeffrey/kojev`. Never bind the server
publicly.

```bash
# 1. sync this checkout to gpu01
./scripts/sync.sh

# 2. CUDA smoke
sbatch scripts/slurm/smoke.sbatch -- python -c 'import torch; print(torch.cuda.get_device_name(0))'

# 3. gold corpus (KoBEST is not a source)
uv run python -m kojev.data_gold --out /data2/jeffrey/kojev/data/gold

# 4. optional AI Hub lane (absent dir raises NotImplementedError)
uv run python -m kojev.data_aihub --root "$AIHUB_DIR"

# 5. teacher labels (ledger-capped)
uv run python -m kojev.label --gold /data2/jeffrey/kojev/data/gold/train.jsonl \
  --out /data2/jeffrey/kojev/data/distill/train.jsonl \
  --ledger /data2/jeffrey/kojev/data/distill/ledger.jsonl

# 6. SFT smoke (interactive partition)
sbatch scripts/slurm/smoke.sbatch -- python -m kojev.train \
  --train /data2/jeffrey/kojev/data/gold/train.jsonl \
  --val /data2/jeffrey/kojev/data/gold/val.jsonl \
  --limit 2000 --epochs 1 --seed 0 --bf16 --batch-size 16 \
  --out /data2/jeffrey/kojev/runs/sft-smoke

# 7. full SFT: 3 seeds + kf-deberta control + distill arm
sbatch scripts/slurm/train.sbatch -- python -m kojev.train --epochs 1 --seed 0 --bf16 --batch-size 16 \
  --out /data2/jeffrey/kojev/runs/sft-full-seed0
# seeds 1 and 2, then:
#   --model kakaobank/kf-deberta-base --out .../sft-full-control-deberta
#   --distill /data2/jeffrey/kojev/data/distill/train.jsonl --out .../sft-full-distill

# 8. summarise (diverged rows flagged, excluded from the median)
uv run python -m kojev.report --runs /data2/jeffrey/kojev/runs

# 9. serve locally, then tear down
KOJEV_CKPT=/data2/jeffrey/kojev/release/kojev-v0 \
  uv run uvicorn kojev.serve:app --host 127.0.0.1 --port 8930
curl -s -X POST http://127.0.0.1:8930/v1/systemone -H 'Content-Type: application/json' \
  --data-binary @tests/fixtures/req.json
kill $SERVER_PID
kill -0 $SERVER_PID   # must fail
```

## Data licenses

Each gold source, as declared on its Hugging Face dataset card (retrieved
2026-09-20). KoBEST is held out and is not listed. AI Hub is an optional
user-supplied lane; we do not redistribute it.

| source | HF card license |
|---|---|
| e9t/nsmc | cc-by-2.0 |
| klue/klue (ynat, nli, sts) | cc-by-sa-4.0 |
| kakaobrain/kor_nli (multi_nli, snli) | cc-by-sa-4.0 |
| jeanlee/kmhas_korean_hate_speech | cc-by-sa-4.0 |
| searle-j/kote | mit |
| wicho/kor_3i4k | cc-by-4.0 |
| lawcompany/KLAID | cc-by-nc-nd-4.0 |
| KorQuAD/squad_kor_v1 | cc-by-nd-4.0 |
| smilegate-ai/kor_unsmile | unspecified on the dataset card |
| AI Hub (optional) | user-supplied; not bundled |

KLAID is CC-BY-NC-ND. The local bundle and this repository do not redistribute
dataset rows; gold JSONL lives under `/data2/jeffrey/kojev/data` and is
gitignored.

## Limitations

- **OOD ceiling.** The typed-decisions recipe expects in-domain gains with a
  much weaker OOD transfer. Do not read val acc as a KoBEST number.
- **Question-count imbalance.** Noul is ~75% of val. Full 104k SFT still
  leaves NSMC near chance on A.X (+0.004 to +0.032). Unsmile jumps to 0.91+
  after the negative-noul fix. DeBERTa on the 512-window subset learns NSMC
  (+0.337), so the data is labeled; A.X at 1 epoch is the weaker backbone.
- **ModernBERT.** The original recipe's ModernBERT instability record
  (loss 1.5→6.1) is why this project uses `skt/A.X-Encoder-base` and a
  divergence watch (running-mean train loss up >25% over 200 steps, or NaN).
  The watch has false-diverged long runs before; the current detector is the
  corrected one in `kojev/train.py`.
- **Teacher labels.** Qwen3-VL-8B via OpenRouter, capped, resumable. Labels
  are not human gold. Distill questions are loss-weighted 0.5. Parser
  failures are dropped, not imputed.
- **Serving.** `POST /v1/systemone` on 127.0.0.1 only. No auth, no public
  bind, no text-generation endpoint. A tampered `kojev_config.json`
  `temperature` string raises `EncodingError` before the backbone loads.
- **Checkpoints.** Median product ckpt is
  `/data2/jeffrey/kojev/runs/sft-full-seed2/checkpoint`. Serving QA in todo 15
  used an earlier trained run; F3 re-curls the median bundle.

## Layout

```
kojev/           library + CLIs (schema, gold, train, serve, release, report)
scripts/slurm/   smoke.sbatch, train.sbatch, rlcd.sbatch, eval.sbatch
tests/           pytest; no KoBEST in training fixtures
eval/RESULTS.md  numbers, with pending tables called pending
data/            gitignored; gold + distill JSONL, OpenRouter ledger
```
