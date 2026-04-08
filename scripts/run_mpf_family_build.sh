#!/usr/bin/env bash
# 依次用不同 grid / 候选数 / mask 配置调用 build_mpf_pipeline.py，生成 benchmark family 数据。
# 与 papers/plan.md 及 experiments/revision_suite.json 中的 benchmark_family 条目对齐。
#
# 用法（在仓库根目录执行）:
#   bash scripts/run_mpf_family_build.sh
#
# 可选环境变量:
#   INPUT_JSON     输入 JSON 列表
#   OUT_BASE       输出根目录（默认: datasets/mpf）
#   SOURCE_DATASET 写入 metadata.source_dataset（默认: coco）
#   SEED           全局随机种子基数（默认: 42）
#   MAX_SAMPLES    若设置，则每个配置只处理前 N 条（调试用）
#   SUBSET_SIZE    随机子集 JSON 的样本数（默认: 1000）
#   NUM_WORKERS    并行数（例如 8）；不设则串行（与旧行为一致）
#   PARALLEL_BACKEND  process（默认，CPU 友好）或 thread（偏 I/O）
#
# 说明: 默认使用 --skip-parquet，只写 images/、任务 JSON、meta JSON，不生成 mpf.parquet。
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/scripts/build_mpf_pipeline.py"

INPUT_JSON="${INPUT_JSON:-${ROOT}/datasets/coco/export/test.json}"
OUT_BASE="${OUT_BASE:-${ROOT}/datasets/mpf_family}"
SOURCE_DATASET="${SOURCE_DATASET:-coco}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SUBSET_SIZE="${SUBSET_SIZE:-1000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "错误: 找不到输入文件: $INPUT_JSON" >&2
  echo "请设置 INPUT_JSON。" >&2
  exit 1
fi

mkdir -p "$OUT_BASE"

run_one() {
  local name="$1"
  shift
  local out_dir="${OUT_BASE}/${name}"
  local image_dir="${out_dir}/images"
  mkdir -p "$out_dir"
  mkdir -p "$image_dir"

  local -a extra=()
  if [[ -n "$MAX_SAMPLES" ]]; then
    extra+=(--max-samples "$MAX_SAMPLES")
  fi
  if [[ -n "$NUM_WORKERS" ]]; then
    extra+=(--num-workers "$NUM_WORKERS" --parallel-backend "$PARALLEL_BACKEND")
  fi

  echo "========== ${name} =========="
  python "$PY" \
    --input-dataset-path "$INPUT_JSON" \
    --input-format json \
    --output-dataset-path "${out_dir}/mpf.parquet" \
    --skip-parquet \
    --image-dir "$image_dir" \
    --task-json-name "${out_dir}/mpf_tasks.json" \
    --subset-json-name "${out_dir}/mpf_tasks_sample_${SUBSET_SIZE}.json" \
    --meta-json-name "${out_dir}/mpf_meta.json" \
    --subset-size "$SUBSET_SIZE" \
    --source-dataset "$SOURCE_DATASET" \
    --seed "$SEED" \
    --prompt-key eval_standard \
    "${extra[@]}" \
    "$@"
  echo ""
}

# revision_suite: bf_g4x4_c4_rect
run_one bf_g4x4_c4_rect \
  --grid-cols 4 --grid-rows 4 \
  --num-candidates 4 \
  --mask-style rect

# revision_suite: bf_g8x6_c4_rect
run_one bf_g8x6_c4_rect \
  --grid-cols 8 --grid-rows 6 \
  --num-candidates 4 \
  --mask-style rect

# revision_suite: bf_g8x8_c8_rect
run_one bf_g8x8_c8_rect \
  --grid-cols 8 --grid-rows 8 \
  --num-candidates 8 \
  --mask-style rect

# revision_suite: bf_g12x12_c16_rect
run_one bf_g12x12_c16_rect \
  --grid-cols 12 --grid-rows 12 \
  --num-candidates 16 \
  --mask-style rect

# revision_suite: bf_g8x6_c4_ellipse
run_one bf_g8x6_c4_ellipse \
  --grid-cols 8 --grid-rows 6 \
  --num-candidates 4 \
  --mask-style ellipse

# 泛化实验 gen_4way_to_8way：评测用 8×6 + 8-way
run_one bf_g8x6_c8_rect \
  --grid-cols 8 --grid-rows 6 \
  --num-candidates 8 \
  --mask-style rect

echo "全部完成。输出目录: $OUT_BASE"
echo "每个子目录: images/ + mpf_tasks.json + 随机子集 JSON + mpf_meta.json（默认不写 mpf.parquet）"
