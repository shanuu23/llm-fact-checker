# Data

```text
data/
├── HASHES.txt                              SHA-256 of the frozen files (see below)
├── fever/
│   ├── raw/                                downloaded separately, NOT in git
│   │   ├── shared_task_dev.jsonl
│   │   └── wiki-pages/wiki-pages/wiki-001.jsonl … wiki-109.jsonl
│   └── processed/fever_baseline_150.jsonl  the frozen sample, in git
└── vitaminc/
    ├── raw/vitaminc/test.jsonl             downloaded separately, NOT in git
    └── processed/vitaminc_sensitivity_100_pairs.jsonl   the frozen sample, in git
```

The two `processed/` files are the inputs to every experiment. You only need `raw/` to re-create
them or to build a different sample. `data/*/raw/` is gitignored because the Wikipedia dump is
several GB.

## Frozen samples

These must not change without a team decision, because every model's results are compared on them.

| File | Rows | SHA-256 |
|---|---|---|
| `fever/processed/fever_baseline_150.jsonl` | 150 | `6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb` |
| `vitaminc/processed/vitaminc_sensitivity_100_pairs.jsonl` | 100 pairs | `731a0ec8510a4e7c22be86afc0378279d644fec5aa4c6d041eb0b787c0155800` |

Check them from the repository root. `HASHES.txt` also lists the raw FEVER dev file, so that line
reports a failure if you have not downloaded the raw data; the other lines are independent.

```bash
shasum -a 256 -c data/HASHES.txt
```

## Raw FEVER

Download from the FEVER dataset page (fever.ai):

1. **Shared Task Development Dataset (Labelled)** to `data/fever/raw/shared_task_dev.jsonl`
   (19,998 claims, 6,666 per label).
2. **Pre-processed Wikipedia Pages (June 2017 dump)**, `wiki-pages.zip` (about 1.7 GB), then:

   ```bash
   mkdir -p data/fever/raw
   unzip -q wiki-pages.zip -d data/fever/raw/wiki-pages -x "__MACOSX/*"
   ls data/fever/raw/wiki-pages/wiki-pages | wc -l      # expect 109
   ```

Verify the downloads reproduce the sample. This writes a separate file and changes nothing:

```bash
uv run factcheck prepare --output data/fever/processed/verify_150.jsonl
shasum -a 256 data/fever/processed/verify_150.jsonl     # expect 6bc7cf91…
```

You do not need the FEVER training set, the blind test set (it has no labels) or the unlabelled
dev file.

| File | SHA-256 |
|---|---|
| `shared_task_dev.jsonl` | `e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7` |
| `wiki-pages.zip` | `4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2` (1,713,485,474 bytes) |

The zip hash is the download that reproduced the frozen sample; it is recorded here, not enforced
by the code.

## Raw VitaminC

Only `test.jsonl` is used (55,197 rows). It is published by the authors
(GitHub `TalSchuster/VitaminC`) and on Hugging Face as `tals/vitaminc`:

```bash
mkdir -p data/vitaminc/raw/vitaminc
curl -L -o data/vitaminc/raw/vitaminc/test.jsonl \
  https://huggingface.co/datasets/tals/vitaminc/resolve/main/test.jsonl
```

Two byte-different exports of the same data exist:

| Source | SHA-256 | Notes |
|---|---|---|
| Original used for the pilot | `7f799e076dec184e31533d46704fef2022d08ef68a0db2c44f138d2b4e552f36` | what the code currently accepts |
| Hugging Face export | `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad` | adds a `big_bench_canary` field to every row |

The Hugging Face file contains identical claims, evidence and labels: it yields the same 20,232
valid pairs and re-selects the same 100 pairs. **The VitaminC loader currently checks for the
pilot hash only, so it rejects the Hugging Face file** until the code is changed to accept both.
This is a pending, deliberate change; do not edit the expected hash by hand.

## Why the hashes matter

The consistency and robustness experiments refuse to run unless the FEVER sample matches its
expected hash, and the VitaminC experiment does the same for the raw test file. The baseline run
does not check or record the sample hash yet (a known gap). A mismatch means the data is not the
data the published numbers came from.
