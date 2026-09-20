# KoJev results (snapshot)

These numbers are **not** the todo 12 five-run table and **not** the todo 14
held-out benchmark. They are the cluster artefacts that exist today, recorded
so the README is not empty while full SFT seeds sit in the Slurm queue behind
`QOSMaxGRESPerUser`.

Recompute from a runs directory with:

```bash
uv run python -m kojev.report --runs /data2/jeffrey/kojev/runs
```

Diverged rows are listed and excluded from the median. That is the selection
rule; it is not a suggestion.

## Smoke (`--limit 2000`, 1 epoch)

| run | acc | majority | gap | brier | ece | diverged |
|---|---:|---:|---:|---:|---:|---|
| sft-smoke-sampled | 0.6290 | 0.6012 | +0.0278 | 0.4995 | 0.1238 | no |
| sft-smoke-clean | 0.6675 | 0.6385 | +0.0290 | 0.4403 | 0.0585 | no |
| sft-smoke-fixed | 0.6551 | 0.6271 | +0.0280 | 0.4547 | 0.0820 | no |
| sft-smoke-gpu6 | 0.7381 | 0.7066 | +0.0315 | 0.3702 | 0.0754 | no |
| sft-smoke-gpu4 | — | — | — | — | — | yes |
| sft-smoke-b8 | — | — | — | — | — | yes |

Todo 10's gate is `val_acc_all - majority_all >= 0.05`. No completed smoke
clears it. The gap is consistently ~+0.03.

## Longer gold-only run (`--limit 20000`, 1 epoch, `sft-20k-truewatch`)

`diverged=false`, temperature 1.15.

| slice | n | acc | majority | gap |
|---|---:|---:|---:|---:|
| overall | 19386 | 0.6738 | 0.6446 | +0.0291 |
| kind:choice | 4349 | 0.3656 | 0.2417 | +0.1239 |
| kind:noul | 14537 | 0.7771 | 0.7853 | −0.0083 |
| kind:score | 500 | 0.3520 | 0.3460 | +0.0060 |

Per source, the same checkpoint is at chance on the balanced tasks:

| source | n | acc | majority | gap |
|---|---:|---:|---:|---:|
| e9t/nsmc | 1000 | 0.5000 | 0.5060 | −0.0060 |
| kakaobrain/kor_nli:multi_nli | 1000 | 0.4950 | 0.5000 | −0.0050 |
| kakaobrain/kor_nli:snli | 1000 | 0.4790 | 0.5000 | −0.0210 |
| klue/klue:nli | 1000 | 0.5040 | 0.5000 | +0.0040 |
| KorQuAD/squad_kor_v1 | 999 | 0.4715 | 0.5005 | −0.0290 |
| smilegate-ai/kor_unsmile | 1387 | 0.3367 | 0.4095 | −0.0728 |
| jeanlee/kmhas_korean_hate_speech | 4500 | 0.8702 | 0.8704 | −0.0002 |
| searle-j/kote | 4500 | 0.9540 | 0.8509 | +0.1031 |
| wicho/kor_3i4k | 1000 | 0.6560 | 0.5510 | +0.1050 |

Noul is 75% of val questions. kmhas and kote are heavily negative (majority
0.87 and 0.96). A head that collapses toward "no" looks strong overall and
learns almost nothing on NSMC / NLI. That is why the 0.05 overall gap has not
moved between the 2k smoke and the 20k run.

## Todo 12 five-run table

Queued, not yet run. Jobs 13685/13686/13687 (A: A.X-Encoder-base, gold only,
seeds 0/1/2), 13692 (B: `kakaobank/kf-deberta-base` control), 13693 (C: gold +
distill weight 0.5). All `PENDING (QOSMaxGRESPerUser)` at the time of this
snapshot.

## Todo 14 held-out benchmarks

Not yet run. KoBEST stays fully held out; it is not in the gold builder.

## RLCD

Not yet run. Verdict will be GO or a plain negative result, not a silent skip.
