from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from tire_ai_pattern.common.exceptions import InputDataError, InputTypeError
from tire_ai_pattern.core.detection.transverse_sipe import detect_transverse_sipes


IMAGE_SIZE = 128
DATASET_SOURCE_ROOT = Path(__file__).parents[3] / "datasets" / "task_transverse_sipe_vis"
DEBUG_BASELINE_ROOT = DATASET_SOURCE_ROOT / "debug_baseline"
RESULT_ROOT = Path(__file__).parents[4] / ".results" / "task_transverse_sipe_vis"
DATASET_RUNTIME_ROOT = RESULT_ROOT / "dataset"
DEBUG_OUTPUT_ROOT = RESULT_ROOT / "debug"
DATASET_EXPECTATIONS: dict[Path, tuple[int, list[float]]] = {
    Path("center_inf/center_one_sipe_angular_lug.png"): (1, [26.3]),
    Path("center_inf/center_no_sipe_v_groove.png"): (0, []),
    Path("center_inf/center_two_sipes_upper_lug.png"): (2, [31.6, 41.8]),
    Path("center_inf/center_one_sipe_with_curved_groove_tail.png"): (1, [54.9]),
    Path("center_inf/center_one_sipe_with_oblique_groove_tail.png"): (1, [111.2]),
    Path("side_inf/side_two_sipes_between_lugs.png"): (2, [15.0, 56.5]),
    Path("side_inf/side_one_sipe_notched_lug.png"): (1, [20.2]),
    Path("side_inf/side_two_sipes_with_zigzag_groove_tail.png"): (2, [29.3, 110.5]),
}
DATASET_IMAGE_RELATIVE_PATHS = sorted(DATASET_EXPECTATIONS, key=lambda path: path.as_posix())


def make_small_image_with_sipes(center_rows: list[int], line_width: int = 1) -> np.ndarray:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    half_width = line_width // 2
    for center_row in center_rows:
        start_row = max(0, center_row - half_width)
        end_row = min(IMAGE_SIZE, start_row + line_width)
        image[start_row:end_row, 12:116] = 0
    return image


def make_small_image_with_vertical_line(center_column: int = 64, line_width: int = 1) -> np.ndarray:
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    half_width = line_width // 2
    start_column = max(0, center_column - half_width)
    end_column = min(IMAGE_SIZE, start_column + line_width)
    image[12:116, start_column:end_column] = 0
    return image


def copy_dataset_image_to_results(relative_image_path: Path) -> Path:
    source_path = DATASET_SOURCE_ROOT / relative_image_path
    rst = {"source_exists": source_path.exists()}
    expect_rst = {"source_exists": True}
    assert rst == expect_rst

    runtime_path = DATASET_RUNTIME_ROOT / relative_image_path
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, runtime_path)
    return runtime_path


def save_debug_image_like_dev(image_path: Path, debug_image: np.ndarray) -> Path:
    image_group = get_debug_image_group(image_path)
    output_dir = DEBUG_OUTPUT_ROOT / image_group
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_debug.png"
    success, buffer = cv2.imencode(".png", debug_image)
    rst = {"encode_success": success}
    expect_rst = {"encode_success": True}
    assert rst == expect_rst
    buffer.tofile(str(output_path))
    return output_path


def get_debug_image_group(image_path: Path) -> str:
    return "center" if image_path.parent.name == "center_inf" else "side"


def get_debug_baseline_path(image_path: Path) -> Path:
    image_group = get_debug_image_group(image_path)
    return DEBUG_BASELINE_ROOT / image_group / f"{image_path.stem}_debug.png"


class TestDetectTransverseSipes:
    @pytest.mark.parametrize("relative_image_path", DATASET_IMAGE_RELATIVE_PATHS, ids=lambda path: path.name)
    def test_dataset_images_match_expected_counts_and_debug_baseline(self, relative_image_path: Path):
        image_path = copy_dataset_image_to_results(relative_image_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            pytest.fail(f"failed to read dataset image: {image_path}")

        expected_count, expected_positions = DATASET_EXPECTATIONS[relative_image_path]
        sipe_count, sipe_positions_px, sipe_widths_px, line_mask, debug_image = detect_transverse_sipes(
            image,
            is_debug=True,
        )
        positions_match = len(sipe_positions_px) == len(expected_positions) and np.allclose(
            sipe_positions_px,
            expected_positions,
            atol=2.0,
        )

        rst = {
            "image_shape": image.shape,
            "sipe_count": sipe_count,
            "positions_match": positions_match,
            "widths_count": len(sipe_widths_px),
            "widths_in_poc_range": all(1 <= width <= 5 for width in sipe_widths_px),
            "line_mask_exists": line_mask is not None,
            "line_mask_shape": line_mask.shape if line_mask is not None else None,
            "debug_image_exists": debug_image is not None,
            "debug_image_shape": debug_image.shape if debug_image is not None else None,
        }
        expect_rst = {
            "image_shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "sipe_count": expected_count,
            "positions_match": True,
            "widths_count": expected_count,
            "widths_in_poc_range": True,
            "line_mask_exists": True,
            "line_mask_shape": (IMAGE_SIZE, IMAGE_SIZE),
            "debug_image_exists": True,
            "debug_image_shape": image.shape,
        }
        assert rst == expect_rst

        debug_output_path = save_debug_image_like_dev(image_path, debug_image)
        rst = {"debug_output_exists": debug_output_path.exists()}
        expect_rst = {"debug_output_exists": True}
        assert rst == expect_rst

        saved_debug_image = cv2.imread(str(debug_output_path), cv2.IMREAD_COLOR)
        if saved_debug_image is None:
            pytest.fail(f"failed to read debug output image: {debug_output_path}")

        baseline_debug_path = get_debug_baseline_path(image_path)
        rst = {"baseline_exists": baseline_debug_path.exists()}
        expect_rst = {"baseline_exists": True}
        assert rst == expect_rst

        baseline_debug_image = cv2.imread(str(baseline_debug_path), cv2.IMREAD_COLOR)
        if baseline_debug_image is None:
            pytest.fail(f"failed to read debug baseline image: {baseline_debug_path}")

        rst = {
            "baseline_debug_shape": baseline_debug_image.shape,
            "debug_matches_baseline": np.array_equal(saved_debug_image, baseline_debug_image),
        }
        expect_rst = {
            "baseline_debug_shape": saved_debug_image.shape,
            "debug_matches_baseline": True,
        }
        assert rst == expect_rst, f"debug image changed: output={debug_output_path}, baseline={baseline_debug_path}"

    def test_image_with_two_sipes_detects_two_lines(self):
        image = make_small_image_with_sipes([40, 86])

        sipe_count, sipe_positions_px, _sipe_widths_px, line_mask, debug_image = detect_transverse_sipes(image)

        rst = {
            "sipe_count": sipe_count,
            "positions_count": len(sipe_positions_px),
            "positions_match": np.allclose(sipe_positions_px, [40.0, 86.0], atol=2.0),
            "line_mask": line_mask,
            "debug_image": debug_image,
        }
        expect_rst = {
            "sipe_count": 2,
            "positions_count": 2,
            "positions_match": True,
            "line_mask": None,
            "debug_image": None,
        }
        assert rst == expect_rst

    def test_two_sipes_only_reports_features(self):
        image = make_small_image_with_sipes([40, 86])

        sipe_count, _sipe_positions_px, sipe_widths_px, _line_mask, _debug_image = detect_transverse_sipes(image)

        rst = {
            "sipe_count": sipe_count,
            "sipe_widths_count": len(sipe_widths_px),
            "widths_in_poc_range": all(1 <= width <= 5 for width in sipe_widths_px),
        }
        expect_rst = {
            "sipe_count": 2,
            "sipe_widths_count": 2,
            "widths_in_poc_range": True,
        }
        assert rst == expect_rst

    def test_small_gap_inside_sipe_is_bridged(self):
        image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
        image[64:65, 12:58] = 0
        image[64:65, 63:116] = 0

        sipe_count, sipe_positions_px, _sipe_widths_px, _line_mask, _debug_image = detect_transverse_sipes(image)

        rst = {
            "sipe_count": sipe_count,
            "positions_match": np.allclose(sipe_positions_px, [64.0], atol=2.0),
        }
        expect_rst = {
            "sipe_count": 1,
            "positions_match": True,
        }
        assert rst == expect_rst

    def test_wide_transverse_groove_is_not_counted_as_sipe(self):
        image = make_small_image_with_sipes([64], line_width=8)

        sipe_count, sipe_positions_px, sipe_widths_px, _line_mask, _debug_image = detect_transverse_sipes(image)

        rst = {
            "sipe_count": sipe_count,
            "sipe_positions_px": sipe_positions_px,
            "sipe_widths_px": sipe_widths_px,
        }
        expect_rst = {
            "sipe_count": 0,
            "sipe_positions_px": [],
            "sipe_widths_px": [],
        }
        assert rst == expect_rst

    def test_thin_line_connected_to_wide_transverse_groove_is_not_counted_as_sipe(self):
        image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
        image[60:69, 12:65] = 0
        image[64:65, 64:116] = 0

        sipe_count, sipe_positions_px, sipe_widths_px, _line_mask, _debug_image = detect_transverse_sipes(image)

        rst = {
            "sipe_count": sipe_count,
            "sipe_positions_px": sipe_positions_px,
            "sipe_widths_px": sipe_widths_px,
        }
        expect_rst = {
            "sipe_count": 0,
            "sipe_positions_px": [],
            "sipe_widths_px": [],
        }
        assert rst == expect_rst

    def test_vertical_line_is_not_counted_as_transverse_sipe(self):
        image = make_small_image_with_vertical_line()

        sipe_count, sipe_positions_px, sipe_widths_px, _line_mask, _debug_image = detect_transverse_sipes(image)

        rst = {
            "sipe_count": sipe_count,
            "sipe_positions_px": sipe_positions_px,
            "sipe_widths_px": sipe_widths_px,
        }
        expect_rst = {
            "sipe_count": 0,
            "sipe_positions_px": [],
            "sipe_widths_px": [],
        }
        assert rst == expect_rst

    def test_edge_residual_is_ignored(self):
        image = make_small_image_with_sipes([5])

        sipe_count, sipe_positions_px, sipe_widths_px, _line_mask, _debug_image = detect_transverse_sipes(image)

        rst = {
            "sipe_count": sipe_count,
            "sipe_positions_px": sipe_positions_px,
            "sipe_widths_px": sipe_widths_px,
        }
        expect_rst = {
            "sipe_count": 0,
            "sipe_positions_px": [],
            "sipe_widths_px": [],
        }
        assert rst == expect_rst

    def test_debug_mode_returns_mask_and_debug_image(self):
        image = make_small_image_with_sipes([64])

        sipe_count, _positions_px, _widths_px, line_mask, debug_image = detect_transverse_sipes(image, is_debug=True)

        rst = {
            "sipe_count": sipe_count,
            "line_mask_exists": line_mask is not None,
            "line_mask_shape": line_mask.shape if line_mask is not None else None,
            "debug_image_exists": debug_image is not None,
            "debug_image_shape": debug_image.shape if debug_image is not None else None,
        }
        expect_rst = {
            "sipe_count": 1,
            "line_mask_exists": True,
            "line_mask_shape": (IMAGE_SIZE, IMAGE_SIZE),
            "debug_image_exists": True,
            "debug_image_shape": image.shape,
        }
        assert rst == expect_rst

    def test_debug_image_marks_segment_without_full_width_guideline(self):
        image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
        image[64:65, 30:70] = 0

        sipe_count, positions_px, _widths_px, _line_mask, debug_image = detect_transverse_sipes(image, is_debug=True)

        center_row = int(round(positions_px[0]))
        green = np.array([0, 255, 0], dtype=np.uint8)
        rst = {
            "sipe_count": sipe_count,
            "left_edge_is_green": bool(np.array_equal(debug_image[center_row, 0], green)),
            "right_edge_is_green": bool(np.array_equal(debug_image[center_row, IMAGE_SIZE - 1], green)),
        }
        expect_rst = {
            "sipe_count": 1,
            "left_edge_is_green": False,
            "right_edge_is_green": False,
        }
        assert rst == expect_rst

    def test_non_bgr_image_raises_input_data_error(self):
        image = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)

        with pytest.raises(InputDataError) as exc_info:
            detect_transverse_sipes(image)

        rst = {"has_shape_message": "shape (H, W, 3)" in str(exc_info.value)}
        expect_rst = {"has_shape_message": True}
        assert rst == expect_rst

    def test_non_array_image_raises_input_type_error(self):
        with pytest.raises(InputTypeError) as exc_info:
            detect_transverse_sipes(None)

        rst = {"has_image_message": "image" in str(exc_info.value)}
        expect_rst = {"has_image_message": True}
        assert rst == expect_rst

    def test_invalid_pixel_parameter_raises_input_data_error(self):
        image = make_small_image_with_sipes([64])

        with pytest.raises(InputDataError) as exc_info:
            detect_transverse_sipes(image, min_width_px=0)

        rst = {"has_min_width_message": "min_width_px" in str(exc_info.value)}
        expect_rst = {"has_min_width_message": True}
        assert rst == expect_rst


class TestTransverseSipeCoverageBranches:
    def test_input_type_branches_are_raised(self):
        image = make_small_image_with_sipes([64])

        with pytest.raises(InputTypeError):
            detect_transverse_sipes(image, is_debug=1)  # type: ignore[arg-type]

    def test_input_relation_branches_are_raised(self):
        image = make_small_image_with_sipes([64])

        with pytest.raises(InputDataError):
            detect_transverse_sipes(image, min_width_px=2, max_width_px=1)

        with pytest.raises(InputDataError):
            detect_transverse_sipes(image, min_width_px=2, narrow_cluster_px=1)

        with pytest.raises(InputDataError):
            detect_transverse_sipes(image, max_angle_deg=85)

    def test_positive_number_and_int_validators_branches(self):
        image = make_small_image_with_sipes([64])

        with pytest.raises(InputTypeError):
            detect_transverse_sipes(image, nominal_width_px=True)  # type: ignore[arg-type]

        with pytest.raises(InputDataError):
            detect_transverse_sipes(image, nominal_width_px=0)

        with pytest.raises(InputTypeError):
            detect_transverse_sipes(image, min_width_px=1.5)  # type: ignore[arg-type]

        with pytest.raises(InputTypeError):
            detect_transverse_sipes(image, edge_margin_px=1.5)  # type: ignore[arg-type]

        with pytest.raises(InputDataError):
            detect_transverse_sipes(image, edge_margin_px=-1)
