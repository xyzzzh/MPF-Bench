# MPF-Bench

MPF-Bench (Masked Patch Finding Benchmark) is a **programmatically verifiable benchmark family** for **fine-grained visual reasoning** in vision-language models (VLMs).

Given a natural image, MPF-Bench partitions it into patches, masks one target patch, and asks the model to identify the candidate patch that correctly fills the missing region. Because each instance is defined directly by construction, MPF-Bench provides:

- deterministic labels
- exact scoring
- zero-cost annotation
- controllable difficulty through grid size, candidate count, and mask shape

## Resources

- Project page: https://xyzzzh.github.io/MPF-Bench/
- Hugging Face dataset: https://huggingface.co/datasets/xyzzzh/MPF-Bench
- Dataset card: `docs/Datasets_card.md`

## Current Release

The current benchmark release contains **6 representative configurations** and **6,000 MPF test instances** in total:

- `bf_g4x4_c4_rect`
- `bf_g8x6_c4_rect`
- `bf_g8x6_c4_ellipse`
- `bf_g8x6_c8_rect`
- `bf_g8x8_c8_rect`
- `bf_g12x12_c16_rect`

These vary along three configurable dimensions:

- **Grid size**
- **Candidate count**
- **Mask shape**

## Repository Structure

```text
MPF-Bench/
├── Paper/                  # paper source and figures
├── assets/                 # website assets
├── docs/
│   └── Datasets_card.md    # Hugging Face dataset README / dataset card
├── scripts/
│   ├── build_mpf_pipeline.py
│   └── run_mpf_family_build.sh
├── index.html              # English project page
└── index-zh.html           # Chinese project page
```

## Quick Start

### 1. Build the benchmark family

Run the benchmark-family build script from the repository root:

```bash
bash scripts/run_mpf_family_build.sh
```

By default, this script writes outputs under:

```text
datasets/mpf_family/
```

Each configuration directory contains:

- `images/`
- `mpf_tasks.json`
- `mpf_tasks_sample_1000.json` (or another subset size if configured)
- `mpf_meta.json`

### 2. Key environment variables

The build script supports the following environment variables:

- `INPUT_JSON`: input JSON list
- `OUT_BASE`: output root directory
- `SOURCE_DATASET`: source dataset name written into metadata
- `SEED`: random seed
- `MAX_SAMPLES`: limit the number of processed examples for debugging
- `SUBSET_SIZE`: subset JSON size
- `NUM_WORKERS`: parallel workers
- `PARALLEL_BACKEND`: `process` or `thread`

Example:

```bash
INPUT_JSON=/path/to/test.json \
OUT_BASE=./datasets/mpf_family \
SOURCE_DATASET=coco \
SUBSET_SIZE=1000 \
NUM_WORKERS=16 \
bash scripts/run_mpf_family_build.sh
```

## Data Format

The build pipeline supports both lightweight task JSON export and richer Hugging Face style export.

Representative fields in the generated data include:

- `instruction`
- `output`
- `images`
- `image_id`
- `source_id`

The richer export pipeline also supports fields such as:

- `composite_image`
- `masked_image`
- `target_patch_image`
- `candidate_images`
- `difficulty`
- `layout_meta`
- `metadata`

See `docs/Datasets_card.md` for a fuller dataset-oriented description.

## What MPF-Bench Measures

MPF-Bench is designed to evaluate whether models can use:

- local texture and appearance compatibility
- spatial continuity
- patch-level contextual reasoning
- fine-grained discrimination under increasing ambiguity

It is intended primarily as an **evaluation benchmark**, while training is treated as a secondary utility enabled by the same programmatic structure.

## Notes

- Source images remain under the licenses of their original datasets, such as COCO and Flickr30K.
- MPF-Bench is centered on controlled, reproducible evaluation of fine-grained visual reasoning rather than broad multimodal competence.
