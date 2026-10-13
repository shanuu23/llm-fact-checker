# Data

```text
data/
├── HASHES.txt                              SHA-256 of the frozen files (see below)
├── fever/
│   ├── raw/                                downloaded separately, NOT in git
│   │   ├── shared_task_dev.jsonl
│   │   └── wiki-pages/wiki-pages/wiki-001.jsonl … wiki-109.jsonl
│   ├── processed/fever_baseline_150.jsonl  the frozen sample, in git
│   ├── prepared/                           perturbed inputs + distractor_audit.md, in git
│   └── full/
│       ├── fever_dev_full.jsonl            the whole dev set (13,332 claims), in git
│       └── prepared/                       perturbed inputs for the whole set, NOT in git
│                                           (rebuilt from the file above in about 12 s)
└── vitaminc/
    ├── raw/vitaminc/test.jsonl             the VitaminC test set (27 MB), in git (CC BY-SA 3.0)
    ├── processed/vitaminc_sensitivity_100_pairs.jsonl   the frozen sample, in git
    └── prepared/pair_audit.md              human-review audit of the pairs, in git
```

`prepared/` holds files derived from the frozen samples (the reordered and distractor evidence the
robustness experiment feeds to every model, and the audits a human reviewed). They do not depend on
a model, so every model's run reads the same files.

The two `processed/` files are the inputs to every experiment. `data/*/raw/` is gitignored because
the Wikipedia dump is several GB.

| You want to | Raw data needed |
|---|---|
| Run the FEVER baseline, consistency or robustness | None: they read only the committed 150-claim sample |
| Run the VitaminC sensitivity experiment | `vitaminc/raw/vitaminc/test.jsonl`, **which is committed**: the run re-validates the committed pairs against it and stops if its hash is not an accepted one |
| Re-create or change a sample (`prepare`, `vitaminc-prepare`) | FEVER dev file and Wikipedia dump, or the VitaminC test file |

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

## The whole FEVER dev set

`data/fever/full/fever_dev_full.jsonl` (7.5 MB, about 1.3 MB in git) holds every SUPPORTS/REFUTES
claim with its evidence text. **It is committed**, so running the whole-set experiments needs no raw
FEVER data and no Wikipedia dump. `factcheck prepare --all` rebuilds it from the raw data and gives
the same bytes. It follows the FEVER data audit's eligibility decision:

| | Frozen 150-case sample | Whole set (`--all`) |
|---|---|---|
| Cases | 150 (75 + 75) | 13,332 (6,666 + 6,666) |
| Repeated claim text | removed (lowest ID kept) | kept |
| Claims flagged by the audit | not marked | kept and marked in `flags` |
| Accented page titles | matched exactly | matched in NFC form |
| Committed, sha256 | yes (`6bc7cf91…`) | yes (`46d03c13…`) |
| Rebuild reproduces the committed file | yes | yes |

The 68 flagged claims (21 SUPPORTED, 47 REFUTED) are the audit's possible label inconsistencies:
the exact claim text appears with a different label (`conflicting_duplicate_label`), or claims that
differ only in case, spacing or Unicode form carry different labels
(`normalized_claim_conflict`). They are not removed and no label is changed.

**A note on the frozen sample.** Its builder matches page titles exactly, so one candidate,
FEVER 95159 ("Ricardo Montalbán was Mexican politician."), was skipped because of an accent
mismatch and replaced by the next candidate in the shuffled order (186995). The replacement is
deterministic and the sample is frozen, so results stay comparable; matching accents is therefore
opt-in and used only for the whole set.

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

Only `test.jsonl` is used (55,197 rows, 28.7 MB, about 2.3 MB in git). **It is committed**
(the Hugging Face export, sha256 `7ad1808d…`), so the sensitivity experiment runs from a clean
clone with no download. It is published by the authors (GitHub `TalSchuster/VitaminC`) and on
Hugging Face as `tals/vitaminc`; to fetch it again:

```bash
mkdir -p data/vitaminc/raw/vitaminc
curl -L -o data/vitaminc/raw/vitaminc/test.jsonl \
  https://huggingface.co/datasets/tals/vitaminc/resolve/main/test.jsonl
```

`.gitignore` keeps the rest of `data/vitaminc/raw/` (the train and dev files, if you download
them) out of git.

Two byte-different exports of the same data exist:

| Source | SHA-256 | Notes |
|---|---|---|
| Original used for the pilot | `7f799e076dec184e31533d46704fef2022d08ef68a0db2c44f138d2b4e552f36` | the release the pilot ran on |
| Hugging Face export | `7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad` | adds a `big_bench_canary` field and an empty-string id to every row |

**Both are accepted.** The loader drops the canary and the empty id and restores the original key
order before anything else sees a row, so the Hugging Face file yields the same 20,232 valid pairs
and re-selects the committed 100 pairs byte for byte (verified, and covered by a test whenever the
raw file is present). Any other hash is rejected with the list of accepted ones.

## Sources and licences

Both datasets are derived from Wikipedia and are redistributed here under their authors' terms:

- **FEVER**: Thorne, Vlachos, Christodoulopoulos and Mittal, *FEVER: a Large-scale Dataset for Fact
  Extraction and VERification*, NAACL-HLT 2018. Claims, labels and Wikipedia evidence sentences;
  see the License link on the FEVER dataset page (fever.ai).
- **VitaminC**: Schuster, Fisch and Barzilay, *Get Your Vitamin C! Robust Fact Verification with
  Contrastive Evidence*, NAACL 2021 (`tals/vitaminc`). Its `LICENSE` states that the annotations
  incorporate material from Wikipedia, are made available under the licence terms of the
  applicable Wikipedia article pages or otherwise CC BY-SA 3.0, and that the synthetic annotations
  are based on FEVER.

CC BY-SA 3.0 requires attribution and that derivatives carry the same licence, so keep this
section if you copy the data files elsewhere.

## Why the hashes matter

The consistency and robustness experiments refuse to run unless the FEVER sample matches its
expected hash, and the VitaminC experiment does the same for the raw test file. The baseline run
records the sample hash in `run_config.json` but does not enforce it. In every case the recorded
hash is what lets a comparison confirm that two models were run on the same cases.
