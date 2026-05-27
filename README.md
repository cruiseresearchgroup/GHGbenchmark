# GHGbenchmark

Code release for **GHGbench: A Unified Multi-Entity, Multi-Task Benchmark for
Carbon Emission Prediction**.

This repository contains the source code, experiment scripts, configuration
files, public documentation, and a lightweight quickstart kit for reproducing
the benchmark pipelines described in the paper.

## Repository layout

- `src/`: reusable benchmark, model, evaluation, and utility code
- `scripts/`: data processing, experiment, baseline, and plotting scripts
- `configs/`: data and model configuration files
- `docs/`: public benchmark documentation and feature summaries
- `quickstart_kit/`: minimal TabPFN quickstart with expected output

## Data access

The repository does not redistribute restricted raw company data. Building
benchmark artefacts and public reconstruction instructions are hosted
separately. See `DATA_ACCESS.md` for details on what is included, what must be
reconstructed from upstream sources, and which data components are distributed
outside this code repository.

## Quickstart

The smallest reproduction path is under `quickstart_kit/`:

```bash
cd quickstart_kit
pip install -r requirements_quickstart.txt
python quickstart_tabpfn.py
```

For the full benchmark, install the main dependencies:

```bash
pip install -r requirements.txt
```

Then use the scripts in `scripts/` together with the configuration files in
`configs/`.

## Citation

If you use this repository, please cite the GHGbench paper.
