---
pretty_name: MPF-Bench
language:
  - en
license: other
task_categories:
  - image-classification
  - visual-question-answering
tags:
  - multimodal
  - vision-language
  - benchmark
  - reasoning
  - evaluation
size_categories:
  - 1K<n<10K
---

# MPF-Bench

## Dataset Summary

MPF-Bench (Masked Patch Finding Benchmark) is a programmatically generated benchmark family for **fine-grained visual reasoning** in vision-language models (VLMs).

Given a natural image, MPF-Bench partitions it into patches, masks one target patch, and asks the model to identify the candidate patch that correctly fills the missing region. Because each instance is defined directly by construction, MPF-Bench provides:

- deterministic labels
- exact scoring
- zero-cost annotation
- controlled difficulty through grid size, candidate count, and mask shape

The benchmark is released as **6 representative configurations**, each with **1,000 MPF test instances**, for a total of **6,000 evaluation examples**.

## What This Dataset Measures

MPF-Bench is designed to evaluate whether a model can use:

- local texture and appearance compatibility
- spatial continuity
- context-dependent patch completion
- fine-grained discrimination under increasing ambiguity

It is intended primarily as a **benchmark for evaluation**, not as a general-purpose instruction-tuning dataset.

## Released Configurations

The current release contains the following benchmark configurations:

- `bf_g4x4_c4_rect`
- `bf_g8x6_c4_rect`
- `bf_g8x6_c4_ellipse`
- `bf_g8x6_c8_rect`
- `bf_g8x8_c8_rect`
- `bf_g12x12_c16_rect`

These vary along three configurable dimensions:

- **Grid size**: e.g. `4x4`, `8x6`, `8x8`, `12x12`
- **Candidate count**: e.g. `4-way`, `8-way`, `16-way`
- **Mask shape**: `rect` or `ellipse`

## Data Format

Depending on the export view, MPF-Bench may appear either as:

1. a lightweight instruction-style JSON format, or
2. a richer Hugging Face dataset format with rendered images and metadata

In the richer HF-native export, key fields include:

- `composite_image`: the full rendered benchmark image shown to the model
- `masked_image`: the main image with the target patch removed
- `target_patch_image`: the ground-truth missing patch
- `candidate_images`: candidate patch images
- `problem`: text description of the task
- `prompts`: prompt variants used for evaluation or teacher supervision
- `solution`: ground-truth patch index
- `solution_idx`: index position within the candidate set
- `candidate_patch_indices`: patch ids used in the composite
- `difficulty`: entropy / similarity / ambiguity statistics
- `layout_meta`: rendering layout metadata for the composite image
- `metadata`: configuration-level and provenance metadata

In the lightweight instruction-style JSON view, examples typically contain fields such as:

- `instruction`
- `input`
- `output`
- `images`
- `image_id`
- `id`
- `source_id`

## Example Task

Each MPF instance follows a fixed composite-image protocol. A representative prompt is:

> You are a professional image analysis expert. Given one masked image and its candidate patches, select the single candidate that best fills the masked region. Judge continuity, texture, geometry, color, and semantic plausibility. Return only the final patch index inside `<mpf>` and `</mpf>`.

The answer is deterministic and can be parsed exactly from the returned patch index.

## How To Load

```python
from datasets import load_dataset

dataset = load_dataset("xyzzzh/MPF-Bench", "bf_g8x6_c4_rect")
split_name = list(dataset.keys())[0]
sample = dataset[split_name][0]

print(split_name)
print(sample.keys())
```

If you work with the raw JSON export instead, load the JSON records directly and read the local image paths referenced by the `images` field.

## Data Source and Provenance

MPF-Bench is constructed from widely used natural-image datasets, including:

- COCO
- Flickr30K

The benchmark instances are generated programmatically from source images after data splitting, so that held-out benchmark test images remain separate from images used in the training study described in the paper/project materials.

## Annotation Process

MPF-Bench does **not** rely on manual question writing or manual answer annotation for benchmark construction.

Instead, labels are generated automatically from the image partition and mask placement procedure. This makes the benchmark:

- scalable
- exactly verifiable
- reproducible across models

## Intended Uses

Recommended uses:

- zero-shot or few-shot evaluation of VLMs
- controlled ablations over ambiguity and local reasoning difficulty
- analysis of fine-grained visual reasoning failure modes

Possible secondary use:

- self-supervised or programmatic training experiments based on exact correctness signals

## Limitations

- MPF-Bench measures a specific capability: patch-level local reasoning under controlled ambiguity. It is **not** a full measure of general multimodal intelligence.
- High performance on MPF-Bench should not be interpreted as broad robustness across all vision-language tasks.
- The dataset is derived from existing natural-image sources and therefore inherits some domain and content biases from those sources.
- The benchmark release is centered on representative configurations rather than exhaustive coverage of all possible MPF settings.

## Licensing and Redistribution Notes

All source images remain under the licenses of their original datasets.

MPF-Bench is intended to distribute benchmark instances and derived metadata for evaluation. Please make sure your use complies with the original dataset terms for COCO, Flickr30K, and any other upstream data sources included in the release pipeline.

## Homepage

- Project page: https://xyzzzh.github.io/MPF-Bench
- Code and generation pipeline: https://github.com/xyzzzh/MPF-Bench
