import argparse
import hashlib
import importlib
import io
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm

PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 256 * 1024 * 1024


def image_to_bytes(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    if hasattr(img, "info") and "icc_profile" in img.info:
        img.info.pop("icc_profile", None)

    pnginfo = PngInfo()
    buf = io.BytesIO()
    img.save(
        buf,
        format="PNG",
        pnginfo=pnginfo,
        icc_profile=None,
        optimize=False,
    )
    return buf.getvalue()


def bytes_to_image(blob: bytes) -> Image.Image:
    return Image.open(io.BytesIO(blob))


def normalize_grid(grid: Tuple[int, int]) -> Tuple[int, int]:
    cols, rows = int(grid[0]), int(grid[1])
    if cols <= 0 or rows <= 0:
        raise ValueError(f"Invalid grid: {grid}")
    return cols, rows


def split_image_to_patches(img: Image.Image, grid: Tuple[int, int]) -> List[Image.Image]:
    cols, rows = normalize_grid(grid)
    width, height = img.size
    patch_w, patch_h = width // cols, height // rows
    patches: List[Image.Image] = []
    for row in range(rows):
        for col in range(cols):
            left, top = col * patch_w, row * patch_h
            patches.append(img.crop((left, top, left + patch_w, top + patch_h)))
    return patches


def patch_bbox(mask_idx: int, grid: Tuple[int, int], img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    cols, rows = normalize_grid(grid)
    width, height = img_size
    patch_w, patch_h = width // cols, height // rows
    row = mask_idx // cols
    col = mask_idx % cols
    left, top = col * patch_w, row * patch_h
    return left, top, left + patch_w, top + patch_h


def patch_row_col(mask_idx: int, grid: Tuple[int, int]) -> Tuple[int, int]:
    cols, _ = normalize_grid(grid)
    return mask_idx // cols, mask_idx % cols


def is_patch_pure_color(
    patch: Image.Image,
    color_std_thresh: float = 6,
    color_mean_black: float = 16,
    color_mean_white: float = 240,
) -> bool:
    arr = np.array(patch)
    if arr.ndim == 3:
        arr = arr[..., :3]
    std = float(np.std(arr))
    mean = float(np.mean(arr))
    return (std < color_std_thresh) or (mean < color_mean_black) or (mean > color_mean_white)


def normalized_gray_entropy(patch: Image.Image, bins: int = 64) -> float:
    arr = np.array(patch.convert("L"), dtype=np.float32)
    hist, _ = np.histogram(arr, bins=bins, range=(0, 255), density=True)
    hist = hist + 1e-8
    entropy = float(-np.sum(hist * np.log(hist)))
    return entropy / math.log(bins)


def patch_richness_score(patch: Image.Image) -> float:
    arr_rgb = np.array(patch, dtype=np.float32)
    std_rgb = float(np.std(arr_rgb))

    arr = np.array(patch.convert("L"), dtype=np.float32)
    lap = (
            -4 * arr
        + np.roll(arr, 1, 0)
        + np.roll(arr, -1, 0)
        + np.roll(arr, 1, 1)
        + np.roll(arr, -1, 1)
    )
    lap_var = float(np.var(lap))
    entropy = normalized_gray_entropy(patch)
    return 0.5 * std_rgb + 0.3 * lap_var + 0.2 * entropy


def choose_mask_index_by_richness(
    patches: Sequence[Image.Image],
    valid_indices: Sequence[int],
    topk: int = 5,
    rng: Optional[random.Random] = None,
) -> Tuple[int, List[Tuple[int, float]]]:
    rng = rng or random.Random()
    scores = [(idx, patch_richness_score(patches[idx])) for idx in valid_indices]
    scores.sort(key=lambda item: item[1], reverse=True)
    pool = [idx for idx, _ in scores[: min(topk, len(scores))]]
    return rng.choice(pool), scores


def histogram_similarity(a: Image.Image, b: Image.Image, bins: int = 32) -> float:
    arr_a = np.array(a.convert("L"), dtype=np.float32)
    arr_b = np.array(b.convert("L"), dtype=np.float32)
    hist_a, _ = np.histogram(arr_a, bins=bins, range=(0, 255), density=True)
    hist_b, _ = np.histogram(arr_b, bins=bins, range=(0, 255), density=True)
    hist_a = hist_a / (np.sum(hist_a) + 1e-8)
    hist_b = hist_b / (np.sum(hist_b) + 1e-8)
    return float(np.minimum(hist_a, hist_b).sum())


def manhattan_distance(idx_a: int, idx_b: int, grid: Tuple[int, int]) -> int:
    row_a, col_a = patch_row_col(idx_a, grid)
    row_b, col_b = patch_row_col(idx_b, grid)
    return abs(row_a - row_b) + abs(col_a - col_b)


def choose_negative_indices(
    n_patches: int,
    mask_idx: int,
    num_candidates: int,
    grid: Tuple[int, int],
    rng: random.Random,
    spatial_mode: str = "global",
    min_manhattan_distance: int = 1,
) -> List[int]:
    if num_candidates < 2:
        raise ValueError("num_candidates must be >= 2")
    neg_needed = num_candidates - 1
    all_indices = [idx for idx in range(n_patches) if idx != mask_idx]
    if neg_needed > len(all_indices):
        raise ValueError(f"Need {neg_needed} negatives but only {len(all_indices)} available")

    candidate_pool = all_indices
    if spatial_mode == "far":
        far_pool = [
            idx
            for idx in all_indices
            if manhattan_distance(idx, mask_idx, grid) >= min_manhattan_distance
        ]
        if len(far_pool) >= neg_needed:
            candidate_pool = far_pool

    return rng.sample(candidate_pool, neg_needed)


def make_candidates(
    patches: Sequence[Image.Image],
    mask_idx: int,
    num_candidates: int,
    grid: Tuple[int, int],
    rng: random.Random,
    spatial_mode: str = "global",
    min_manhattan_distance: int = 1,
) -> Tuple[List[bytes], List[int], int]:
    neg_indices = choose_negative_indices(
        n_patches=len(patches),
        mask_idx=mask_idx,
        num_candidates=num_candidates,
        grid=grid,
        rng=rng,
        spatial_mode=spatial_mode,
        min_manhattan_distance=min_manhattan_distance,
    )
    candidate_ids = [mask_idx] + neg_indices
    candidates = [image_to_bytes(patches[idx]) for idx in candidate_ids]
    perm = list(range(len(candidate_ids)))
    rng.shuffle(perm)

    shuffled_candidates = [candidates[idx] for idx in perm]
    shuffled_ids = [candidate_ids[idx] for idx in perm]
    solution_idx = shuffled_ids.index(mask_idx)
    return shuffled_candidates, shuffled_ids, solution_idx


def _get_font(px: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, px)
        except OSError:
            continue
            return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
            return font.getsize(text)


def infer_candidate_layout(num_candidates: int, candidate_cols: Optional[int]) -> Tuple[int, int]:
    cols = candidate_cols or min(4, num_candidates)
    cols = max(1, min(cols, num_candidates))
    rows = int(math.ceil(num_candidates / cols))
    return cols, rows


def apply_mask(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    mask_style: str,
    mask_fill: Tuple[int, int, int],
) -> None:
    if mask_style == "rect":
        draw.rectangle(bbox, fill=mask_fill)
        return
    if mask_style == "ellipse":
        draw.ellipse(bbox, fill=mask_fill)
        return
    if mask_style == "rounded_rect":
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(bbox, radius=8, fill=mask_fill)
        else:
            draw.rectangle(bbox, fill=mask_fill)
        return
    raise ValueError(f"Unsupported mask_style: {mask_style}")


def build_masked_image(
        img: Image.Image,
    grid: Tuple[int, int],
    mask_idx: int,
    mask_style: str,
    mask_fill: Tuple[int, int, int],
) -> Image.Image:
    masked = img.copy()
    draw = ImageDraw.Draw(masked)
    apply_mask(draw, patch_bbox(mask_idx, grid, img.size), mask_style, mask_fill)
    return masked


def build_composite_image_single(
    img: Image.Image,
    grid: Tuple[int, int],
    mask_idx: int,
    candidates: Sequence[bytes],
    candidate_patch_indices: Sequence[int],
    candidate_cols: Optional[int] = None,
    gap: int = 6,
    margin: int = 6,
    mask_fill: Tuple[int, int, int] = (180, 180, 180),
    mask_style: str = "rect",
    cand_ann_color: Tuple[int, int, int] = (0, 0, 255),
    cand_font_px: int = 16,
) -> Tuple[Image.Image, Dict[str, object], Image.Image]:
    if len(candidates) != len(candidate_patch_indices):
        raise ValueError("candidates and candidate_patch_indices must have the same length")

    cols, rows = normalize_grid(grid)
    width, height = img.size
    patch_w, patch_h = width // cols, height // rows
    masked = build_masked_image(img, grid, mask_idx, mask_style, mask_fill)

    candidate_imgs = [bytes_to_image(blob).convert("RGB") for blob in candidates]
    font = _get_font(cand_font_px)
    label_bar = cand_font_px + 6
    per_tile_w, per_tile_h = patch_w, patch_h + label_bar
    layout_cols, layout_rows = infer_candidate_layout(len(candidates), candidate_cols)

    canvas_w = max(width, layout_cols * per_tile_w + gap * (layout_cols - 1)) + margin * 2
    canvas_h = height + gap + layout_rows * per_tile_h + gap * max(layout_rows - 1, 0) + margin * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))

    main_x, main_y = margin, margin
    canvas.paste(masked, (main_x, main_y))

    grid_x = margin
    grid_y = main_y + height + gap
    draw = ImageDraw.Draw(canvas)
    candidate_bboxes = []

    for idx, (tile, patch_id) in enumerate(zip(candidate_imgs, candidate_patch_indices)):
        row = idx // layout_cols
        col = idx % layout_cols
        x = grid_x + col * (per_tile_w + gap)
        y = grid_y + row * (per_tile_h + gap)
        canvas.paste(tile, (x, y))
        label = str(patch_id)
        text_w, text_h = _text_size(draw, label, font)
        text_x = x + (per_tile_w - text_w) // 2
        text_y = y + patch_h + (label_bar - text_h) // 2
        draw.text((text_x, text_y), label, fill=cand_ann_color, font=font)
        candidate_bboxes.append([x, y, x + per_tile_w, y + per_tile_h])

    layout_meta = {
        "canvas_size": [canvas_w, canvas_h],
        "main_bbox": [main_x, main_y, main_x + width, main_y + height],
        "mask_index": int(mask_idx),
        "grid": list(grid),
        "patch_size": [patch_w, patch_h],
        "mask_bbox": list(patch_bbox(mask_idx, grid, img.size)),
        "mask_style": mask_style,
        "candidates": {
            "cols": layout_cols,
            "rows": layout_rows,
            "tile_size": [per_tile_w, per_tile_h],
            "start_xy": [grid_x, grid_y],
            "gap": gap,
            "candidate_bboxes": candidate_bboxes,
        },
        "fonts": {
            "mask_px": 0,
            "cand_px": cand_font_px,
        },
    }
    return canvas, layout_meta, masked


def border_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def compute_shortcut_scores(
    img: Image.Image,
    grid: Tuple[int, int],
    mask_idx: int,
    candidate_images: Sequence[Image.Image],
) -> Dict[str, object]:
    left, top, right, bottom = patch_bbox(mask_idx, grid, img.size)
    img_arr = np.array(img.convert("RGB"))

    context_edges = {}
    if top > 0:
        context_edges["top"] = img_arr[top - 1:top, left:right]
    if bottom < img_arr.shape[0]:
        context_edges["bottom"] = img_arr[bottom:bottom + 1, left:right]
    if left > 0:
        context_edges["left"] = img_arr[top:bottom, left - 1:left]
    if right < img_arr.shape[1]:
        context_edges["right"] = img_arr[top:bottom, right:right + 1]

    ring_bands = list(context_edges.values())
    ring_mean = None
    if ring_bands:
        flat_ring = np.concatenate([band.reshape(-1, 3) for band in ring_bands], axis=0)
        ring_mean = flat_ring.mean(axis=0)

    boundary_scores: List[float] = []
    ring_color_scores: List[float] = []
    combined_scores: List[float] = []

    for candidate in candidate_images:
        arr = np.array(candidate.convert("RGB"))
        edge_scores = []
        if "top" in context_edges:
            edge_scores.append(-border_mse(arr[:1, :, :], context_edges["top"]))
        if "bottom" in context_edges:
            edge_scores.append(-border_mse(arr[-1:, :, :], context_edges["bottom"]))
        if "left" in context_edges:
            edge_scores.append(-border_mse(arr[:, :1, :], context_edges["left"]))
        if "right" in context_edges:
            edge_scores.append(-border_mse(arr[:, -1:, :], context_edges["right"]))

        boundary_score = float(np.mean(edge_scores)) if edge_scores else 0.0
        boundary_scores.append(boundary_score)

        if ring_mean is None:
            ring_score = 0.0
        else:
            ring_score = float(-np.mean(np.abs(arr.reshape(-1, 3).mean(axis=0) - ring_mean)))
        ring_color_scores.append(ring_score)
        combined_scores.append(boundary_score + 0.5 * ring_score)

    return {
        "boundary_scores": boundary_scores,
        "ring_color_scores": ring_color_scores,
        "combined_scores": combined_scores,
        "boundary_argmax": int(np.argmax(boundary_scores)),
        "ring_argmax": int(np.argmax(ring_color_scores)),
        "combined_argmax": int(np.argmax(combined_scores)),
    }


def compute_difficulty(
    patches: Sequence[Image.Image],
    mask_idx: int,
    candidate_patch_indices: Sequence[int],
    shortcut_scores: Dict[str, object],
) -> Dict[str, float]:
    target_patch = patches[mask_idx]
    target_entropy = normalized_gray_entropy(target_patch)
    negative_indices = [idx for idx in candidate_patch_indices if idx != mask_idx]
    negative_similarities = [histogram_similarity(target_patch, patches[idx]) for idx in negative_indices]
    best_negative_similarity = max(negative_similarities) if negative_similarities else 0.0

    solution_position = candidate_patch_indices.index(mask_idx)
    boundary_scores = shortcut_scores["boundary_scores"]
    best_negative_boundary = max(
        score for idx, score in enumerate(boundary_scores) if idx != solution_position
    ) if len(boundary_scores) > 1 else boundary_scores[solution_position]
    boundary_margin = float(abs(boundary_scores[solution_position] - best_negative_boundary))
    boundary_ambiguity = 1.0 / (1.0 + boundary_margin)
    difficulty_score = 0.4 * target_entropy + 0.4 * best_negative_similarity + 0.2 * boundary_ambiguity

    return {
        "target_entropy": float(target_entropy),
        "best_negative_similarity": float(best_negative_similarity),
        "boundary_ambiguity": float(boundary_ambiguity),
        "difficulty_score": float(difficulty_score),
    }


def build_prompt_bundle(answer_patch_id: int, num_candidates: int) -> Dict[str, str]:
    eval_standard = (
        "You are a professional image analysis expert. Given one masked image and its candidate patches, "
        f"select the single candidate that best fills the masked region from {num_candidates} choices. "
        "Judge continuity, texture, geometry, color, and semantic plausibility. "
        "Return your reasoning inside <think></think> and the final patch index inside <mpf></mpf>."
    )
    eval_concise = (
        "Choose the patch index that best restores the masked region. "
        "Use visual continuity and local context only. "
        "Answer with <mpf>patch_index</mpf>."
    )
    eval_reason_first = (
        "First infer what should appear inside the masked region, then compare every candidate patch against that prediction. "
        f"There are {num_candidates} candidates. Return a compact explanation in <think></think> and the final numeric patch index in <mpf></mpf>."
    )
    teacher_cot = (
        "You are generating high-quality chain-of-thought data for a visual completion task. "
        "Analyze the masked region using surrounding visual cues, discuss each candidate by its displayed index, "
        f"and conclude with the correct patch index {answer_patch_id}. "
        "Your response must use the format <think>...</think><mpf>{answer_patch_id}</mpf>."
    )
    return {
        "eval_standard": eval_standard,
        "eval_concise": eval_concise,
        "eval_reason_first": eval_reason_first,
        "teacher_cot": teacher_cot,
    }


def _coerce_metadata_int(value: object, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_image(example: Dict[str, object]) -> Image.Image:
    image_obj = example.get("image")
    if image_obj is None:
        image_obj = example.get("image_path")

    if isinstance(image_obj, str):
        path = image_obj
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        return bytes_to_image(image_obj["bytes"]).convert("RGB")
    if isinstance(image_obj, bytes):
        return bytes_to_image(image_obj).convert("RGB")
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    raise ValueError(
        f"Unsupported image format: {type(image_obj)}. "
        "Expected 'image' (bytes/dict/PIL) or 'image_path' (str)."
    )


def per_example_seed_for_parallel(base_seed: int, idx: int) -> int:
    """
    并行模式下每条样本使用独立种子，结果与调度顺序无关、可复现。
    （与 num_workers=1 时主线程里连续 randint 的分布不同。）
    """
    digest = hashlib.sha256(f"{int(base_seed)}:{int(idx)}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _example_to_plain_dict(example: object) -> Dict[str, object]:
    if isinstance(example, dict):
        return dict(example)
    if hasattr(example, "as_dict"):
        return dict(example.as_dict())  # type: ignore[arg-type]
    try:
        return dict(example)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError(f"Cannot convert example to dict: {type(example)}")


def _process_pool_initializer() -> None:
    """避免「多进程 × 每进程多线程 BLAS」导致 CPU 过订阅、变慢。"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _process_mpf_job(
    job: Tuple[
        int,
        Dict[str, object],
        Tuple[int, int],
        int,
        int,
        Tuple[int, int],
        bool,
        float,
        float,
        float,
        int,
        str,
        int,
        str,
        str,
        Optional[int],
        str,
    ],
) -> Tuple[int, Optional[Dict[str, object]]]:
    (
        idx,
        example,
        grid,
        num_candidates,
        local_seed,
        img_resize,
        avoid_pure_mask,
        color_std_thresh,
        color_mean_black,
        color_mean_white,
        richness_topk,
        spatial_mode,
        min_manhattan_distance,
        mask_style,
        prompt_key,
        candidate_cols,
        source_dataset,
    ) = job
    row = process_example_single(
        example=example,
        grid=grid,
        num_candidates=num_candidates,
        seed=local_seed,
        img_resize=img_resize,
        avoid_pure_mask=avoid_pure_mask,
        color_std_thresh=color_std_thresh,
        color_mean_black=color_mean_black,
        color_mean_white=color_mean_white,
        richness_topk=richness_topk,
        spatial_mode=spatial_mode,
        min_manhattan_distance=min_manhattan_distance,
        mask_style=mask_style,
        prompt_key=prompt_key,
        candidate_cols=candidate_cols,
        source_dataset=source_dataset,
    )
    return idx, row


def extract_source_id(example: Dict[str, object]) -> str:
    """Prefer COCO-style image_id so MPF rows align with original image id."""
    for key in ("image_id", "id", "file_name", "filename", "path"):
        if key in example and example[key] is not None:
            return str(example[key])
    return "unknown"


def extract_source_image_path(example: Dict[str, object]) -> Optional[str]:
    for key in ("image_path", "path", "file_name", "filename"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_source_image_name(example: Dict[str, object]) -> str:
    source_path = extract_source_image_path(example)
    if source_path:
        return os.path.basename(source_path)
    return extract_source_id(example)


def sanitize_filename_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    safe = safe.strip("._")
    return safe or "unknown"


def extract_source_image_stem(example: Dict[str, object]) -> str:
    source_name = extract_source_image_name(example)
    stem, _ = os.path.splitext(source_name)
    if not stem:
        stem = extract_source_id(example)
    return sanitize_filename_component(stem)


def build_output_image_name(row: Dict[str, object], example: Dict[str, object], dataset_idx: int) -> str:
    source_stem = extract_source_image_stem(example)
    metadata = row.get("metadata", {})
    coco_row_id = _coerce_metadata_int(
        metadata.get("coco_row_id") if isinstance(metadata, dict) else None,
        -1,
    )
    unique_idx = coco_row_id if coco_row_id >= 0 else int(dataset_idx)
    return f"mpf_{source_stem}_{unique_idx:08d}.png"


def save_row_composite_image(
    row: Dict[str, object],
    example: Dict[str, object],
    image_dir: str,
    dataset_idx: int,
) -> Tuple[str, str]:
    image_name = build_output_image_name(row, example, dataset_idx)
    abs_path = os.path.abspath(os.path.join(image_dir, image_name))
    bytes_to_image(row["composite_image"]).convert("RGB").save(abs_path, format="PNG")
    row["image_path"] = abs_path
    return image_name, abs_path


def strip_binary_payload(row: Dict[str, object]) -> None:
    row.pop("composite_image", None)
    row.pop("masked_image", None)
    row.pop("target_patch_image", None)
    row.pop("candidate_images", None)


def build_meta_json_record(
    row: Dict[str, object],
    example: Dict[str, object],
    dataset_idx: int,
    image_name: str,
) -> Dict[str, object]:
    return {
        "dataset_idx": int(dataset_idx),
        "image_name": image_name,
        "image_path": row.get("image_path", ""),
        "source_image_name": extract_source_image_name(example),
        "source_image_stem": extract_source_image_stem(example),
        "source_image_path": extract_source_image_path(example),
        "mask_index": row["mask_index"],
        "mask_row": row["mask_row"],
        "mask_col": row["mask_col"],
        "candidate_patch_indices": row["candidate_patch_indices"],
        "solution": row["solution"],
        "solution_idx": row["solution_idx"],
        "problem": row["problem"],
        "prompts": row["prompts"],
        "metadata": row["metadata"],
        "difficulty": row["difficulty"],
        "shortcut_scores": row["shortcut_scores"],
        "layout_meta": row["layout_meta"],
    }


def build_skip_parquet_exports(
    row: Dict[str, object],
    example: Dict[str, object],
    dataset_idx: int,
    image_dir: str,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, Dict[str, str]]]:
    image_name, abs_path = save_row_composite_image(row, example, image_dir, dataset_idx)
    task_record = build_sft_json_record(row)
    meta_record = build_meta_json_record(row, example, dataset_idx, image_name)
    prompt_mapping = {abs_path: row["prompts"]}
    strip_binary_payload(row)
    return task_record, meta_record, prompt_mapping


def _process_mpf_export_job(
    job: Tuple[
        int,
        Dict[str, object],
        Tuple[int, int],
        int,
        int,
        Tuple[int, int],
        bool,
        float,
        float,
        float,
        int,
        str,
        int,
        str,
        str,
        Optional[int],
        str,
        str,
    ],
) -> Tuple[int, Optional[Dict[str, object]], Optional[Dict[str, object]], Dict[str, Dict[str, str]]]:
    (
        idx,
        example,
        grid,
        num_candidates,
        local_seed,
        img_resize,
        avoid_pure_mask,
        color_std_thresh,
        color_mean_black,
        color_mean_white,
        richness_topk,
        spatial_mode,
        min_manhattan_distance,
        mask_style,
        prompt_key,
        candidate_cols,
        source_dataset,
        image_dir,
    ) = job
    row = process_example_single(
        example=example,
        grid=grid,
        num_candidates=num_candidates,
        seed=local_seed,
        img_resize=img_resize,
        avoid_pure_mask=avoid_pure_mask,
        color_std_thresh=color_std_thresh,
        color_mean_black=color_mean_black,
        color_mean_white=color_mean_white,
        richness_topk=richness_topk,
        spatial_mode=spatial_mode,
        min_manhattan_distance=min_manhattan_distance,
        mask_style=mask_style,
        prompt_key=prompt_key,
        candidate_cols=candidate_cols,
        source_dataset=source_dataset,
    )
    if row is None:
        return idx, None, None, {}
    task_record, meta_record, prompt_mapping = build_skip_parquet_exports(
        row=row,
        example=example,
        dataset_idx=idx,
        image_dir=image_dir,
    )
    return idx, task_record, meta_record, prompt_mapping


def process_example_single(
    example: Dict[str, object],
    grid: Tuple[int, int] = (8, 6),
    num_candidates: int = 4,
    seed: int = 42,
    img_resize: Tuple[int, int] = (640, 480),
    avoid_pure_mask: bool = True,
    color_std_thresh: float = 6,
    color_mean_black: float = 16,
    color_mean_white: float = 240,
    richness_topk: int = 5,
    spatial_mode: str = "global",
    min_manhattan_distance: int = 1,
    mask_style: str = "rect",
    prompt_key: str = "eval_standard",
    candidate_cols: Optional[int] = None,
    source_dataset: str = "unknown",
) -> Optional[Dict[str, object]]:
    rng = random.Random(seed)
    img = resolve_image(example)
    if img.size != img_resize:
        img = img.resize(img_resize, Image.BILINEAR)

    patches = split_image_to_patches(img, grid=grid)
    valid_indices = list(range(len(patches)))
    if avoid_pure_mask:
        valid_indices = [
            idx for idx, patch in enumerate(patches)
            if not is_patch_pure_color(patch, color_std_thresh, color_mean_black, color_mean_white)
        ]
        if not valid_indices:
            return None

    mask_idx, richness_ranking = choose_mask_index_by_richness(
        patches, valid_indices, topk=richness_topk, rng=rng
    )
    candidates, candidate_ids, solution_idx = make_candidates(
        patches=patches,
        mask_idx=mask_idx,
        num_candidates=num_candidates,
        grid=grid,
        rng=rng,
        spatial_mode=spatial_mode,
        min_manhattan_distance=min_manhattan_distance,
    )

    composite_img, layout_meta, masked_img = build_composite_image_single(
        img=img,
        grid=grid,
        mask_idx=mask_idx,
        candidates=candidates,
        candidate_patch_indices=candidate_ids,
        candidate_cols=candidate_cols,
        mask_style=mask_style,
    )

    candidate_pil = [bytes_to_image(blob).convert("RGB") for blob in candidates]
    shortcut_scores = compute_shortcut_scores(img, grid, mask_idx, candidate_pil)
    difficulty = compute_difficulty(patches, mask_idx, candidate_ids, shortcut_scores)
    correct_patch_id = int(candidate_ids[solution_idx])
    prompts = build_prompt_bundle(answer_patch_id=correct_patch_id, num_candidates=num_candidates)
    if prompt_key not in prompts:
        raise ValueError(f"Unknown prompt_key: {prompt_key}")

    mask_row, mask_col = patch_row_col(mask_idx, grid)
    source_id = extract_source_id(example)

    row = {
        "composite_image": image_to_bytes(composite_img),
        "masked_image": image_to_bytes(masked_img),
        "target_patch_image": image_to_bytes(patches[mask_idx]),
        "candidate_images": [image_to_bytes(image) for image in candidate_pil],
        "image_path": "",
        "mask_index": int(mask_idx),
        "mask_row": int(mask_row),
        "mask_col": int(mask_col),
        "candidate_patch_indices": [int(value) for value in candidate_ids],
        "solution": correct_patch_id,
        "solution_idx": int(solution_idx),
        "problem": prompts[prompt_key],
        "prompts": prompts,
        "layout_meta": layout_meta,
        "difficulty": difficulty,
        "shortcut_scores": {
            "boundary_scores": [float(value) for value in shortcut_scores["boundary_scores"]],
            "ring_color_scores": [float(value) for value in shortcut_scores["ring_color_scores"]],
            "combined_scores": [float(value) for value in shortcut_scores["combined_scores"]],
            "boundary_argmax": int(shortcut_scores["boundary_argmax"]),
            "ring_argmax": int(shortcut_scores["ring_argmax"]),
            "combined_argmax": int(shortcut_scores["combined_argmax"]),
        },
        "metadata": {
            "mask_num": 1,
            "patch_grid": [int(grid[0]), int(grid[1])],
            "img_size": [int(img.size[0]), int(img.size[1])],
            "n_candidates": int(num_candidates),
            "mask_style": mask_style,
            "prompt_key": prompt_key,
            "source_dataset": source_dataset,
            "source_id": source_id,
            "coco_image_id": _coerce_metadata_int(example.get("image_id"), -1),
            "coco_row_id": _coerce_metadata_int(example.get("id"), -1),
            "seed": int(seed),
            "candidate_layout": [layout_meta["candidates"]["cols"], layout_meta["candidates"]["rows"]],
            "richness_rank_top_score": float(richness_ranking[0][1]),
        },
    }
    return row


def resolve_output_path(base_dir: Optional[str], name_or_path: str) -> str:
    if os.path.isabs(name_or_path):
        return name_or_path
    if base_dir is None:
        return os.path.abspath(name_or_path)
    return os.path.abspath(os.path.join(base_dir, name_or_path))


def ensure_parquet_path(path: str) -> str:
    return path if path.endswith(".parquet") else f"{path}.parquet"


def build_sft_json_record(row: Dict[str, object]) -> Dict[str, object]:
    image_path = str(row.get("image_path", "")).strip()
    if not image_path:
        raise ValueError("Cannot export SFT-style JSON without image_path.")

    prompt = str(row.get("problem", "")).strip()
    instruction = prompt if prompt.startswith("<image>") else f"<image>\n{prompt}"
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    record: Dict[str, object] = {
        "instruction": instruction,
        "input": "",
        "output": f"<mpf>{int(row['solution'])}</mpf>",
        "images": [image_path],
    }

    coco_image_id = _coerce_metadata_int(metadata.get("coco_image_id"), -1)
    coco_row_id = _coerce_metadata_int(metadata.get("coco_row_id"), -1)
    if coco_image_id >= 0:
        record["image_id"] = coco_image_id
    if coco_row_id >= 0:
        record["id"] = coco_row_id

    source_id = metadata.get("source_id")
    if source_id is not None:
        record["source_id"] = str(source_id)
    return record


def export_sft_json_files(
    task_records: Sequence[Dict[str, object]],
    task_json_path: str,
    subset_json_path: Optional[str],
    subset_size: int,
    seed: int,
) -> None:
    os.makedirs(os.path.dirname(task_json_path) or ".", exist_ok=True)
    with open(task_json_path, "w", encoding="utf-8") as handle:
        json.dump(list(task_records), handle, ensure_ascii=False, indent=2)

    if subset_json_path is None or subset_size <= 0:
        return

    records_list = list(task_records)
    sample_size = min(int(subset_size), len(records_list))
    sampled = (
        random.Random(seed).sample(records_list, sample_size)
        if sample_size < len(records_list)
        else records_list
    )
    os.makedirs(os.path.dirname(subset_json_path) or ".", exist_ok=True)
    with open(subset_json_path, "w", encoding="utf-8") as handle:
        json.dump(sampled, handle, ensure_ascii=False, indent=2)


def export_meta_json_file(meta_records: Sequence[Dict[str, object]], meta_json_path: str) -> None:
    os.makedirs(os.path.dirname(meta_json_path) or ".", exist_ok=True)
    with open(meta_json_path, "w", encoding="utf-8") as handle:
        json.dump(list(meta_records), handle, ensure_ascii=False, indent=2)


def load_json_records(json_path: str) -> List[Dict[str, object]]:
    """
    Load a list of records from export JSON.
    Resolves relative image_path against the JSON file directory.
    Expected fields per record: image_id, id, caption, image_path, ...
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"JSON must be a list of records, got {type(data)}")
    json_dir = os.path.dirname(os.path.abspath(json_path))
    out: List[Dict[str, object]] = []
    for rec in data:
        row = dict(rec)
        if "image_path" in row and isinstance(row["image_path"], str):
            path = row["image_path"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(json_dir, path))
            row["image_path"] = path
        out.append(row)
    return out


def build_features():
    datasets_module = importlib.import_module("datasets")
    Features = datasets_module.Features
    HFSequence = datasets_module.Sequence
    Value = datasets_module.Value

    return Features(
        {
        "composite_image": Value("binary"),
            "masked_image": Value("binary"),
            "target_patch_image": Value("binary"),
            "candidate_images": HFSequence(Value("binary")),
            "image_path": Value("string"),
        "mask_index": Value("int32"),
            "mask_row": Value("int32"),
            "mask_col": Value("int32"),
            "candidate_patch_indices": HFSequence(Value("int32")),
            "solution": Value("int32"),
            "solution_idx": Value("int32"),
        "problem": Value("string"),
            "prompts": {
                "eval_standard": Value("string"),
                "eval_concise": Value("string"),
                "eval_reason_first": Value("string"),
                "teacher_cot": Value("string"),
            },
        "layout_meta": {
                "canvas_size": HFSequence(Value("int32")),
                "main_bbox": HFSequence(Value("int32")),
            "mask_index": Value("int32"),
                "grid": HFSequence(Value("int32")),
                "patch_size": HFSequence(Value("int32")),
                "mask_bbox": HFSequence(Value("int32")),
                "mask_style": Value("string"),
            "candidates": {
                "cols": Value("int32"),
                "rows": Value("int32"),
                    "tile_size": HFSequence(Value("int32")),
                    "start_xy": HFSequence(Value("int32")),
                    "gap": Value("int32"),
                    "candidate_bboxes": HFSequence(HFSequence(Value("int32"))),
            },
            "fonts": {
                "mask_px": Value("int32"),
                    "cand_px": Value("int32"),
                },
            },
            "difficulty": {
                "target_entropy": Value("float32"),
                "best_negative_similarity": Value("float32"),
                "boundary_ambiguity": Value("float32"),
                "difficulty_score": Value("float32"),
            },
            "shortcut_scores": {
                "boundary_scores": HFSequence(Value("float32")),
                "ring_color_scores": HFSequence(Value("float32")),
                "combined_scores": HFSequence(Value("float32")),
                "boundary_argmax": Value("int32"),
                "ring_argmax": Value("int32"),
                "combined_argmax": Value("int32"),
        },
        "metadata": {
            "mask_num": Value("int32"),
                "patch_grid": HFSequence(Value("int32")),
                "img_size": HFSequence(Value("int32")),
                "n_candidates": Value("int32"),
                "mask_style": Value("string"),
                "prompt_key": Value("string"),
                "source_dataset": Value("string"),
                "source_id": Value("string"),
                "coco_image_id": Value("int64"),
                "coco_row_id": Value("int64"),
                "seed": Value("int64"),
                "candidate_layout": HFSequence(Value("int32")),
                "richness_rank_top_score": Value("float32"),
            },
        }
    )


def process_hf_dataset_single(
    input_dataset_path: str,
    output_dataset_path: str,
    split: str = "test",
    grid: Tuple[int, int] = (8, 6),
    num_candidates: int = 4,
    seed: int = 42,
    img_resize: Tuple[int, int] = (640, 480),
    avoid_pure_mask: bool = True,
    color_std_thresh: float = 6,
    color_mean_black: float = 16,
    color_mean_white: float = 240,
    richness_topk: int = 5,
    image_dir: Optional[str] = None,
    mapping_json_name: Optional[str] = None,
    meta_json_name: Optional[str] = None,
    source_dataset: str = "unknown",
    spatial_mode: str = "global",
    min_manhattan_distance: int = 1,
    mask_style: str = "rect",
    prompt_key: str = "eval_standard",
    candidate_cols: Optional[int] = None,
    max_samples: Optional[int] = None,
    input_format: Optional[str] = None,
    task_json_name: Optional[str] = None,
    subset_json_name: Optional[str] = None,
    subset_size: int = 1000,
    skip_parquet: bool = False,
    num_workers: int = 1,
    parallel_backend: str = "process",
) -> str:
    num_workers = max(1, int(num_workers))
    parallel_backend = (parallel_backend or "process").lower()
    if parallel_backend not in {"process", "thread"}:
        raise ValueError(f"parallel_backend must be 'process' or 'thread', got {parallel_backend!r}")

    rng = random.Random(seed)
    output_path = ensure_parquet_path(output_dataset_path)
    output_base_dir = os.path.dirname(os.path.abspath(output_path))
    mapping_json_path = (
        resolve_output_path(output_base_dir, mapping_json_name)
        if mapping_json_name
        else None
    )
    meta_json_path = (
        resolve_output_path(output_base_dir, meta_json_name)
        if meta_json_name
        else None
    )
    task_json_path = (
        resolve_output_path(output_base_dir, task_json_name)
        if task_json_name is not None
        else None
    )
    subset_json_path = (
        resolve_output_path(output_base_dir, subset_json_name)
        if subset_json_name is not None
        else None
    )
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
    if mapping_json_path is not None:
        os.makedirs(os.path.dirname(mapping_json_path) or ".", exist_ok=True)
    if meta_json_path is not None:
        os.makedirs(os.path.dirname(meta_json_path) or ".", exist_ok=True)
    if task_json_path is not None and not image_dir:
        raise ValueError("--task-json-name requires --image-dir so exported JSON can reference the composite image.")
    if meta_json_path is not None and not image_dir:
        raise ValueError("--meta-json-name requires --image-dir so meta records can reference the composite image.")
    if num_workers > 1 and not skip_parquet:
        print("Warning: parallel mode with parquet output falls back to serial to avoid large IPC of image bytes.")
        num_workers = 1

    if input_format is None or input_format == "auto":
        input_format = (
            "json" if input_dataset_path.lower().endswith(".json") else "parquet"
        )

    # JSON 输入且不写 parquet 时不需要 HuggingFace datasets（避免依赖与额外内存）。
    need_datasets_lib = (input_format != "json") or (not skip_parquet)
    datasets_module = None
    Dataset = None
    if need_datasets_lib:
        datasets_module = importlib.import_module("datasets")
        Dataset = datasets_module.Dataset

    if input_format == "json":
        records = load_json_records(input_dataset_path)
        if max_samples is not None:
            records = records[: max_samples]
        dataset_iter: object = records
        num_rows = len(records)
    else:
        load_dataset = datasets_module.load_dataset
        dataset_dict = load_dataset("parquet", data_files={split: input_dataset_path})
        dataset = dataset_dict[split]
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        dataset_iter = dataset
        num_rows = len(dataset)

    rows: List[Dict[str, object]] = []
    task_records: List[Tuple[int, Dict[str, object]]] = []
    meta_records: List[Tuple[int, Dict[str, object]]] = []
    path_to_prompt: Dict[str, Dict[str, str]] = {}
    print(
        f"Input: {input_format} ({input_dataset_path})\n"
        f"Processing {num_rows} images "
        f"(grid={grid[0]}x{grid[1]}, candidates={num_candidates}, mask={mask_style}, prompt={prompt_key}"
        f"{'' if num_workers <= 1 else f', workers={num_workers} ({parallel_backend})'}) ..."
    )

    if num_workers <= 1:
        for idx, example in enumerate(tqdm(dataset_iter, desc="Processing examples", total=num_rows)):
            local_seed = rng.randint(0, 1 << 30)
            row = process_example_single(
                example=example,
                grid=grid,
                num_candidates=num_candidates,
                seed=local_seed,
                img_resize=img_resize,
                avoid_pure_mask=avoid_pure_mask,
                color_std_thresh=color_std_thresh,
                color_mean_black=color_mean_black,
                color_mean_white=color_mean_white,
                richness_topk=richness_topk,
                spatial_mode=spatial_mode,
                min_manhattan_distance=min_manhattan_distance,
                mask_style=mask_style,
                prompt_key=prompt_key,
                candidate_cols=candidate_cols,
                source_dataset=source_dataset,
            )
            if row is None:
                continue
            plain_example = _example_to_plain_dict(example)
            if image_dir:
                image_name, abs_path = save_row_composite_image(row, plain_example, image_dir, idx)
                if mapping_json_path is not None:
                    path_to_prompt[abs_path] = row["prompts"]
                if task_json_path is not None:
                    task_records.append((idx, build_sft_json_record(row)))
                if meta_json_path is not None:
                    meta_records.append((idx, build_meta_json_record(row, plain_example, idx, image_name)))
            if skip_parquet:
                strip_binary_payload(row)
            else:
                rows.append(row)
    else:
        if input_format == "json":
            indexed_examples: List[Tuple[int, Dict[str, object]]] = [
                (i, _example_to_plain_dict(records[i])) for i in range(len(records))
            ]
        else:
            print(
                "注意: 并行模式会将 HuggingFace Dataset 一次性 materialize 到内存 (to_list())，"
                "大数据集可能占用大量 RAM。"
            )
            plain_list = dataset.to_list()
            indexed_examples = [(i, _example_to_plain_dict(plain_list[i])) for i in range(len(plain_list))]

        jobs: List[Tuple[int, Dict[str, object], Tuple[int, int], int, int, Tuple[int, int], bool, float, float, float, int, str, int, str, str, Optional[int], str, str]] = []
        for idx, ex in indexed_examples:
            local_seed = per_example_seed_for_parallel(seed, idx)
            jobs.append(
                (
                    idx,
                    ex,
                    grid,
                    num_candidates,
                    local_seed,
                    img_resize,
                    avoid_pure_mask,
                    color_std_thresh,
                    color_mean_black,
                    color_mean_white,
                    richness_topk,
                    spatial_mode,
                    min_manhattan_distance,
                    mask_style,
                    prompt_key,
                    candidate_cols,
                    source_dataset,
                    image_dir or "",
                )
            )

        if parallel_backend == "process":
            pool = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_process_pool_initializer,
            )
        else:
            pool = ThreadPoolExecutor(max_workers=num_workers)
        with pool:
            future_to_idx = {
                pool.submit(_process_mpf_export_job, job): job[0]
                for job in jobs
            }
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Processing examples"):
                idx, task_record, meta_record, prompt_mapping = future.result()
                if task_record is None or meta_record is None:
                    continue
                task_records.append((idx, task_record))
                meta_records.append((idx, meta_record))
                path_to_prompt.update(prompt_mapping)

    task_records_sorted = [record for _idx, record in sorted(task_records, key=lambda item: item[0])]
    meta_records_sorted = [record for _idx, record in sorted(meta_records, key=lambda item: item[0])]

    if mapping_json_path is not None:
        with open(mapping_json_path, "w", encoding="utf-8") as handle:
            json.dump(path_to_prompt, handle, ensure_ascii=False, indent=2)

    if task_json_path is not None:
        export_sft_json_files(
            task_records=task_records_sorted,
            task_json_path=task_json_path,
            subset_json_path=subset_json_path,
            subset_size=subset_size,
            seed=seed,
        )
    if meta_json_path is not None:
        export_meta_json_file(meta_records_sorted, meta_json_path)

    if not skip_parquet:
        assert Dataset is not None
        out_dataset = Dataset.from_list(rows, features=build_features())
        out_dataset.to_parquet(output_path)
        print(f"Done. Samples: {len(out_dataset)}")
        print(f"Saved Parquet: {output_path}")
    else:
        sample_count = len(task_records_sorted) if task_json_path is not None else len(meta_records_sorted)
        print(f"Done. Samples: {sample_count} (Parquet skipped)")

    if image_dir:
        print(f"Saved images to: {os.path.abspath(image_dir)}")
    if mapping_json_path is not None:
        print(f"Saved prompt mapping JSON: {mapping_json_path}")
    if task_json_path is not None:
        print(f"Saved task JSON: {task_json_path}")
    if subset_json_path is not None and subset_size > 0:
        print(f"Saved sampled task JSON: {subset_json_path}")
    if meta_json_path is not None:
        print(f"Saved meta JSON: {meta_json_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MPF benchmark-family instances from parquet or JSON (e.g. coco export test.json)."
    )
    parser.add_argument(
        "--input-dataset-path",
        required=True,
        help="Input parquet path/glob, or JSON list export (e.g. datasets/coco/export/test.json).",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "parquet", "json"],
        default="auto",
        help="Input format. auto: .json -> json, else parquet.",
    )
    parser.add_argument(
        "--output-dataset-path",
        required=True,
        help="Output parquet path (with or without .parquet), or any path under the desired output directory when using --skip-parquet.",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Do not write Parquet; only export images / task JSON / meta JSON (saves memory, no HuggingFace datasets needed for JSON input).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--candidate-cols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=480)
    parser.add_argument("--richness-topk", type=int, default=5)
    parser.add_argument("--spatial-mode", choices=["global", "far"], default="global")
    parser.add_argument("--min-manhattan-distance", type=int, default=1)
    parser.add_argument("--mask-style", choices=["rect", "ellipse", "rounded_rect"], default="rect")
    parser.add_argument(
        "--prompt-key",
        choices=["eval_standard", "eval_concise", "eval_reason_first", "teacher_cot"],
        default="eval_standard",
    )
    parser.add_argument("--image-dir", default=None)
    parser.add_argument(
        "--mapping-json-name",
        default=None,
        help="Optional prompt-mapping JSON path. Leave unset to skip.",
    )
    parser.add_argument(
        "--meta-json-name",
        default=None,
        help="Optional meta JSON export containing layout/difficulty/shortcut/metadata fields.",
    )
    parser.add_argument(
        "--task-json-name",
        default=None,
        help="Optional JSON export with instruction/input/output/images format.",
    )
    parser.add_argument(
        "--subset-json-name",
        default=None,
        help="Optional random subset JSON export derived from --task-json-name.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
        help="Number of samples for --subset-json-name.",
    )
    parser.add_argument("--source-dataset", default="unknown")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-pure-mask", action="store_true", help="Do not filter low-information mask regions.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行 worker 数量；1 为串行（与原 Random 链一致）。>1 时每条样本使用独立哈希种子，结果与串行不完全相同。",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["process", "thread"],
        default="process",
        help="process：多进程，适合 CPU 密集的 PIL/NumPy；thread：多线程，主要对读图 I/O 略有帮助。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_format = None if args.input_format == "auto" else args.input_format
    process_hf_dataset_single(
        input_dataset_path=args.input_dataset_path,
        output_dataset_path=args.output_dataset_path,
        split=args.split,
        grid=(args.grid_cols, args.grid_rows),
        num_candidates=args.num_candidates,
        seed=args.seed,
        img_resize=(args.img_width, args.img_height),
        avoid_pure_mask=not args.allow_pure_mask,
        richness_topk=args.richness_topk,
        image_dir=args.image_dir,
        mapping_json_name=args.mapping_json_name,
        meta_json_name=args.meta_json_name,
        source_dataset=args.source_dataset,
        spatial_mode=args.spatial_mode,
        min_manhattan_distance=args.min_manhattan_distance,
        mask_style=args.mask_style,
        prompt_key=args.prompt_key,
        candidate_cols=args.candidate_cols,
        max_samples=args.max_samples,
        input_format=input_format,
        task_json_name=args.task_json_name,
        subset_json_name=args.subset_json_name,
        subset_size=args.subset_size,
        skip_parquet=args.skip_parquet,
        num_workers=args.num_workers,
        parallel_backend=args.parallel_backend,
    )


if __name__ == "__main__":
    main()