# KoJev: run-report module + README (todos 12/16 partial)

## Why this increment

Todo 12's QA surface is `uv run python -m kojev.report --runs runs/` producing
a table of runs with diverged rows flagged and excluded from the median.
The module did not exist (RED: ModuleNotFoundError: No module named kojev.report).

Todo 16's README was a 3-line stub. The gpu01 round-trip still waits on the
median-seed checkpoint, so todo 16 stays UNCHECKED. This fills every README
section the plan names, with pending tables labelled pending rather than
invented.

## RED

    $ uv run pytest tests/test_report.py -q
    E   ModuleNotFoundError: No module named 'kojev.report'
    !!!! Interrupted: 1 error during collection !!!!
    exit 2

## Fix

kojev/report.py:
- load */report.json under --runs
- markdown table with a DIVERGED status column
- median_accuracy() uses only non-diverged rows
- missing reports / missing overall.accuracy raise ReportError

README.md: what KoJev is, recipe provenance (independent of TypeSafe Jev),
results pointer to eval/RESULTS.md, seed-spread placeholder, RLCD-not-run,
budget audit $2.945571 vs $100 matching the ledger sum, reproduction
commands, licenses table from HF dataset cards retrieved 2026-09-20,
limitations (noul imbalance, ModernBERT record, teacher caveats, local serve).

eval/RESULTS.md: honest snapshot of existing cluster reports. Documents that
the 0.05 smoke gap is not met (~+0.03) because noul (75% of val) sits at or
below majority on kmhas/kote-imbalanced families, and NSMC/NLI are at chance.

## GREEN

    uv run pytest tests/test_report.py -q  => 4 passed
    uv run pytest -q                       => 143 passed
    ruff check / format                    => clean
    basedpyright kojev tests               => 0 errors, 0 warnings

## Mutation

Restoring `healthy = [row.accuracy for row in rows]` (keeping diverged rows)
fails the median test:

    E   assert 0.73 == 0.72 ± 7.2e-07

0.73 is the median of {0.71, 0.73, 0.99}; 0.72 is the median of the two
healthy seeds. The test binds the exclusion rule, not merely the presence of
a "DIVERGED" string.

## Real CLI on cluster report.json files (10 runs)

    sft-20k-stable          0.6747  DIVERGED
    sft-20k-truewatch       0.6738  ok
    sft-budget-20k          0.6739  DIVERGED
    sft-smoke-b8            0.7377  DIVERGED
    sft-smoke-clean         0.6675  ok
    sft-smoke-fixed         0.6551  ok
    sft-smoke-gpu4          0.7384  DIVERGED
    sft-smoke-gpu6          0.7381  ok
    sft-smoke-plan-lr       0.7397  ok
    sft-smoke-sampled       0.6290  ok
    {"median_accuracy": 0.67065, "healthy": 6, "diverged": 4, "runs": 10}

## Budget

    uv run python -m kojev.release audit data/distill/ledger.jsonl
    {"total_usd": 2.9455706409999483, "records": 37557, "under_cap": true}

README quotes this exact total.

## Cluster-side progress (no GPU yet)

Distill labels+ledger rsynced to /data2/jeffrey/kojev/data/distill/.
Todo 12 arms queued:

    13685 A seed 0 gold-only
    13686 A seed 1 gold-only
    13687 A seed 2 gold-only
    13692 B kakaobank/kf-deberta-base control
    13693 C gold + distill weight 0.5

All PENDING QOSMaxGRESPerUser. Todos 10/12/16 remain unchecked: the 0.05
smoke gap is not met, five report.json files do not exist, gpu01 bundle
round-trip is not done.
