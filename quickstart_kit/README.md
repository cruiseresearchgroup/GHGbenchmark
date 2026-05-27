# GHGbench Quickstart Kit

Reproduce the headline TabPFN~v2 result on the 26-city core+climate
grouped-building tier in **one command**.

## What this is

A self-contained, ~50 MB drop-in package that lets reviewers verify
GHGbench's main building-track claim (TabPFN v2 ~ R^2 = 0.479 on unseen
buildings) without setting up the full pipeline.

The kit ships:

| File | Purpose |
| ---- | ------- |
| `quickstart_tabpfn.py` | self-contained 100-line TabPFN v2 reproduction |
| `requirements_quickstart.txt` | minimal pip deps (numpy, pandas, sklearn, torch, tabpfn) |
| `quickstart_data/building_quickstart.csv` (~47 MB) | canonical 471k-row panel with climate joined |
| `quickstart_data/splits_grouped_seed42.npz` (~0.7 MB) | precomputed train/val/test row indices for the paper's stratified grouped-building split |
| `expected_output.txt` | full stdout from a reference run |

`quickstart_data/` is hosted separately (size); see the parent
`paper_prep/dataset_release/landing_page.md` for the download URL.

## One-command reproduction

```bash
pip install -r requirements_quickstart.txt
python quickstart_tabpfn.py --data_dir quickstart_data
```

## Expected output

```
========================================================
  TabPFN v2 | core_all_cities (+climate) | seed=42
========================================================
  R^2     = 0.4508    (paper 3-seed mean: 0.479 +/- 0.024)
  MAE     = 377.7
  n_test  = 94,875
  runtime = ~140s on cuda
========================================================
```

The `R^2 = 0.4508` is the **single-seed-42 row** of the 3-seed mean
reported in Table 2 of the paper. The other two seeds (123, 456) and
their mean / std are produced by the full repository (see
`scripts/run_task_a_tabpfn.py` in the parent directory).

## Runtime

| Hardware | Time |
| -------- | ---- |
| RTX A5000 (24 GB) | ~140 s |
| Any modern NVIDIA GPU >= 8 GB | < 5 min |
| CPU (modern x86) | ~25 min (TabPFN does in-context inference per chunk) |

The script auto-detects CUDA; pass `--device cpu` to force CPU.

## What it actually does

1. Loads `building_quickstart.csv` (471,070 rows, 26 cities, target
   `total_ghg_emissions_mtco2e` in tCO2e).
2. Loads precomputed train / val / test row indices from
   `splits_grouped_seed42.npz`. These were produced by the paper's
   stratified grouped-building splitter (stratify on (city,
   property_type), group on building_id, seed=42), so reviewers
   reproduce exactly the same test rows as the paper.
3. Subsamples 10,000 train rows (matches the paper's TabPFN protocol),
   median-imputes the six features, and fits TabPFN v2.
4. Predicts on all 94,875 test rows in 4096-row chunks (TabPFN holds
   the train set in GPU memory; chunking caps peak GPU memory).
5. Reports R^2 and MAE on the test rows; clips predictions to
   `[0, 2 * max(y_train))]` to match paper protocol.

## Going beyond the quickstart

To run the full benchmark (other models, all six T2 feature tiers,
T1 company track, multimodal Sentinel-2 fusion, paired bootstrap,
forecasting), see the top-level `README.md` and the scripts under
`../scripts/`.
