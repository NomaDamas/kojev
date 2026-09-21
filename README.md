# KoJev

Korean typed-decision encoder: a state plus `choice` / `score` / `noul`
questions in, probabilities out, one forward pass. Independent of TypeSafe
Jev; the wire shape matches `POST /v1/systemone`.

- Backbone: `skt/A.X-Encoder-base` (Apache-2.0), full finetune
- Weights: [`NomaDamas/KoJev-v0`](https://huggingface.co/NomaDamas/KoJev-v0)
- KoBEST held out of training. RLCD attempted, **NO-GO**; this release is SFT-only

## Benchmark

Same 80 examples per split, seed 0. KoJev = this v0 checkpoint.
OpenJev = `com-kotobalabs/open-jev-deberta-v3-large`.
Laya = `convaiinnovations/laya-multilingual`.
Jev = OpenRouter `typesafe/jev-1.13` (480 calls, $0.009).

| split | KoJev | OpenJev | Laya | Jev 1.13 |
|---|---:|---:|---:|---:|
| gold-val | **0.702** | 0.621 | 0.682 | **0.769** |
| boolq | 0.500 | 0.738 | 0.625 | **0.988** |
| copa | 0.538 | 0.675 | 0.500 | **0.988** |
| wic | 0.463 | 0.513 | 0.500 | **0.888** |
| hellaswag | 0.338 | 0.375 | 0.375 | **0.775** |
| sentineg | 0.525 | 0.838 | 0.700 | **0.938** |

Full gold-val (not the 80-sample slice): overall **0.764** vs majority 0.645.
OOD rule-gold **0.416** vs majority 0.482. KoBEST zero-shot is near chance.
In-domain fit is real; generalization is not. Full tables: [eval/RESULTS.md](eval/RESULTS.md).

## Train

Gold mix: 12 Korean HF sources, ~104k train states / ~311k questions.
SFT 1 epoch × 3 seeds (0.7145 / 0.7215 / 0.7211), then one continue-train
epoch from seed 2. Teacher labels from `qwen/qwen3-vl-8b-instruct` via
OpenRouter, $2.95 of a $100 cap.

## Serve

```bash
KOJEV_CKPT=/path/to/KoJev-v0 \
  uv run uvicorn kojev.serve:app --host 127.0.0.1 --port 8930
curl -s -X POST http://127.0.0.1:8930/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @tests/fixtures/req.json
```

Load weights with `kojev.encoder.load_checkpoint`, not
`AutoModelForSequenceClassification`.

## Licenses

KoJev-v0 is Apache-2.0, derived from `skt/A.X-Encoder-base` (Apache-2.0).
Gold JSONL is not in this repo. Source cards: NSMC cc-by-2.0; KLUE / KorNLI /
KMHAS cc-by-sa-4.0; KOTE MIT; 3i4K cc-by-4.0; KLAID cc-by-nc-nd-4.0;
KorQuAD cc-by-nd-4.0; UnSmile unspecified on the dataset card.

## Layout

```
kojev/            library + CLIs
scripts/slurm/    smoke / train / eval
tests/
eval/RESULTS.md
```
