from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tire_ai_pattern.common.exceptions import InputDataError, InputTypeError
from tire_ai_pattern.utils.logger import get_logger


TrackData = list[tuple[int, float, float]]
Segment = tuple[float, float, int, int]
WideClusterMap = dict[int, list[tuple[int, int]]]

logger = get_logger(__name__)


@dataclass
class _ActiveTrack:
    """Temporary track used while following a horizontal sipe column by column."""

    data: TrackData
    last_column: int
    last_center_y: float


def detect_transverse_sipes(
    image: np.ndarray,
    nominal_width_px: float = 0.6,
    min_width_px: int = 1,
    max_width_px: int = 5,
    narrow_cluster_px: int = 5,
    edge_margin_px: int = 13,
    min_segment_length_px: int = 16,
    max_angle_deg: float = 30.0,
    is_debug: bool = False,
) -> tuple[int, list[float], list[float], np.ndarray | None, np.ndarray | None]:
    """
    Detect transverse sipes in a small tire pattern image.

    The core layer only reports image features. It does not know the image
    region, validate Rule9 limits, calculate scores, or save debug artifacts.

    Args:
        image: BGR image with shape ``(H, W, 3)``.
        nominal_width_px: POC nominal sipe width in pixels. Defaults to 0.6.
        min_width_px: Minimum accepted per-column average line width.
        max_width_px: Maximum accepted per-column average line width.
        narrow_cluster_px: Maximum per-column narrow foreground cluster width.
        edge_margin_px: Top/bottom rows ignored as edge residuals.
        min_segment_length_px: Minimum horizontal segment length.
        max_angle_deg: Maximum local deviation from horizontal.
        is_debug: Return line mask and debug overlay when true.

    Returns:
        ``(count, positions_px, widths_px, line_mask, debug_image)``.
    """
    _validate_inputs(
        image=image,
        nominal_width_px=nominal_width_px,
        min_width_px=min_width_px,
        max_width_px=max_width_px,
        narrow_cluster_px=narrow_cluster_px,
        edge_margin_px=edge_margin_px,
        min_segment_length_px=min_segment_length_px,
        max_angle_deg=max_angle_deg,
        is_debug=is_debug,
    )
    logger.debug("start transverse sipe detection")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    binary_image = cv2.adaptiveThreshold(
        blurred_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=5,
    )

    dedup_distance_px = max(1.0, nominal_width_px * 2.0)
    positions, sipe_count, line_mask, widths = _analyze_horizontal_lines(
        binary=binary_image,
        min_width_px=min_width_px,
        narrow_cluster_px=narrow_cluster_px,
        edge_margin_px=edge_margin_px,
        min_segment_length_px=min_segment_length_px,
        max_angle_deg=max_angle_deg,
        max_width_px=max_width_px,
        dedup_distance_px=dedup_distance_px,
    )

    result_mask = None
    debug_image = None
    if is_debug:
        result_mask = line_mask
        debug_image = _draw_debug_image(
            image=image,
            line_mask=line_mask,
            count=sipe_count,
        )

    logger.debug("transverse sipe detection complete, count=%d, positions=%s", sipe_count, positions)
    return sipe_count, positions, widths, result_mask, debug_image


def _validate_inputs(
    image: np.ndarray,
    nominal_width_px: float,
    min_width_px: int,
    max_width_px: int,
    narrow_cluster_px: int,
    edge_margin_px: int,
    min_segment_length_px: int,
    max_angle_deg: float,
    is_debug: bool,
) -> None:
    if not isinstance(image, np.ndarray):
        raise InputTypeError("detect_transverse_sipes", "image", "np.ndarray", type(image).__name__)
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputDataError("detect_transverse_sipes", "image", "expected BGR image with shape (H, W, 3)", image.shape)
    if not isinstance(is_debug, bool):
        raise InputTypeError("detect_transverse_sipes", "is_debug", "bool", type(is_debug).__name__)

    _validate_positive_number("nominal_width_px", nominal_width_px)
    _validate_positive_int("min_width_px", min_width_px)
    _validate_positive_int("max_width_px", max_width_px)
    _validate_positive_int("narrow_cluster_px", narrow_cluster_px)
    _validate_non_negative_int("edge_margin_px", edge_margin_px)
    _validate_positive_int("min_segment_length_px", min_segment_length_px)
    _validate_positive_number("max_angle_deg", max_angle_deg)

    if max_width_px < min_width_px:
        raise InputDataError("detect_transverse_sipes", "max_width_px", "must be greater than or equal to min_width_px", max_width_px)
    if narrow_cluster_px < min_width_px:
        raise InputDataError("detect_transverse_sipes", "narrow_cluster_px", "must be greater than or equal to min_width_px", narrow_cluster_px)
    if max_angle_deg >= 85:
        raise InputDataError("detect_transverse_sipes", "max_angle_deg", "must be less than 85", max_angle_deg)


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_positive_number(param: str, value: object) -> None:
    if not _is_real_number(value):
        raise InputTypeError("detect_transverse_sipes", param, "int or float", type(value).__name__)
    if value <= 0:
        raise InputDataError("detect_transverse_sipes", param, "must be positive", value)


def _validate_positive_int(param: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputTypeError("detect_transverse_sipes", param, "int", type(value).__name__)
    if value <= 0:
        raise InputDataError("detect_transverse_sipes", param, "must be positive", value)


def _validate_non_negative_int(param: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputTypeError("detect_transverse_sipes", param, "int", type(value).__name__)
    if value < 0:
        raise InputDataError("detect_transverse_sipes", param, "must be non-negative", value)


def _bridge_small_horizontal_gaps(binary: np.ndarray, max_gap_px: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap_px + 1, 1))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _split_track_data_by_angle(
    track_data: TrackData,
    max_angle_deg: float,
    smooth_half_window: int = 3,
) -> list[TrackData]:
    if not track_data:
        return []
    if len(track_data) == 1:
        return [track_data]

    center_values = np.array([track_item[1] for track_item in track_data], dtype=np.float64)
    smoothed_centers = np.array(
        [
            center_values[max(0, index - smooth_half_window): min(len(track_data), index + smooth_half_window + 1)].mean()
            for index in range(len(track_data))
        ]
    )

    max_slope = float(np.tan(np.radians(max_angle_deg)))
    segments: list[TrackData] = []
    segment_start = 0

    for column_index in range(1, len(track_data)):
        previous_column = track_data[column_index - 1][0]
        current_column = track_data[column_index][0]
        column_gap = max(1, current_column - previous_column)
        center_delta = abs(smoothed_centers[column_index] - smoothed_centers[column_index - 1])

        if center_delta > max_slope * column_gap:
            segments.append(track_data[segment_start:column_index])
            segment_start = column_index

    segments.append(track_data[segment_start:])
    return [segment for segment in segments if segment]


def _build_sipe_tracks(
    all_column_clusters: list[tuple[int, list[tuple[int, int]]]],
    max_dy: float = 3.0,
    max_gap_columns: int = 5,
) -> list[TrackData]:
    active_tracks: list[_ActiveTrack] = []
    finished_tracks: list[TrackData] = []

    for column_index, clusters in all_column_clusters:
        cluster_info = [((start_row + end_row) / 2.0, float(end_row - start_row + 1)) for start_row, end_row in clusters]

        still_active: list[_ActiveTrack] = []
        for track in active_tracks:
            if column_index - track.last_column > max_gap_columns:
                finished_tracks.append(track.data)
            else:
                still_active.append(track)
        active_tracks = still_active

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(active_tracks):
            for cluster_index, (center_y, _column_width) in enumerate(cluster_info):
                distance = abs(center_y - track.last_center_y)
                if distance <= max_dy:
                    candidates.append((distance, track_index, cluster_index))
        candidates.sort()

        matched_tracks: set[int] = set()
        matched_clusters: set[int] = set()
        for _distance, track_index, cluster_index in candidates:
            if track_index in matched_tracks or cluster_index in matched_clusters:
                continue
            center_y, column_width = cluster_info[cluster_index]
            active_tracks[track_index].data.append((column_index, center_y, column_width))
            active_tracks[track_index].last_column = column_index
            active_tracks[track_index].last_center_y = center_y
            matched_tracks.add(track_index)
            matched_clusters.add(cluster_index)

        for cluster_index, (center_y, column_width) in enumerate(cluster_info):
            if cluster_index not in matched_clusters:
                active_tracks.append(_ActiveTrack(data=[(column_index, center_y, column_width)], last_column=column_index, last_center_y=center_y))

    for track in active_tracks:
        finished_tracks.append(track.data)
    return finished_tracks


def _analyze_horizontal_lines(
    binary: np.ndarray,
    min_width_px: int,
    narrow_cluster_px: int,
    edge_margin_px: int = 0,
    min_segment_length_px: int = 1,
    max_angle_deg: float = 30.0,
    max_width_px: int = 3,
    dedup_distance_px: float = 1.2,
) -> tuple[list[float], int, np.ndarray, list[float]]:
    working_binary = binary.copy()
    image_height = binary.shape[0]

    if edge_margin_px > 0:
        working_binary[:edge_margin_px, :] = 0
        working_binary[max(0, image_height - edge_margin_px):, :] = 0

    wide_groove_min_width_px = max(max_width_px + 3, narrow_cluster_px + 3)
    wide_clusters_by_column = _collect_wide_clusters_by_column(
        binary=working_binary,
        min_width_px=wide_groove_min_width_px,
    )
    bridged_binary = _bridge_small_horizontal_gaps(working_binary, max_gap_px=5)
    max_tilt_horizontal_span = int(min_width_px / np.tan(np.radians(max_angle_deg)))
    horizontal_open_width = max(3, min(max_tilt_horizontal_span, min_segment_length_px // 2))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_open_width, 1))
    bridged_binary = cv2.morphologyEx(bridged_binary, cv2.MORPH_OPEN, open_kernel)

    label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (bridged_binary > 0).astype(np.uint8), connectivity=8
    )

    line_mask = np.zeros_like(binary)
    raw_segments: list[Segment] = []

    for label_id in range(1, label_count):
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        bbox_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bbox_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if bbox_width < min_segment_length_px:
            continue

        all_column_clusters: list[tuple[int, list[tuple[int, int]]]] = []
        for column_index in range(left, left + bbox_width):
            component_rows = np.where(labels[top: top + bbox_height, column_index] == label_id)[0]
            if len(component_rows) == 0:
                continue

            column_clusters = _split_rows_into_clusters(component_rows, top)
            narrow_clusters = [
                (start_row, end_row)
                for start_row, end_row in column_clusters
                if (end_row - start_row + 1) <= narrow_cluster_px
            ]
            if narrow_clusters:
                all_column_clusters.append((column_index, narrow_clusters))

        if not all_column_clusters:
            continue

        tracks = _build_sipe_tracks(all_column_clusters, max_dy=narrow_cluster_px, max_gap_columns=12)
        for track_data in tracks:
            for segment in _split_track_data_by_angle(track_data, max_angle_deg):
                accepted_segment = _validate_segment(
                    segment=segment,
                    min_width_px=min_width_px,
                    max_width_px=max_width_px,
                    min_segment_length_px=min_segment_length_px,
                )
                if accepted_segment is None:
                    continue
                if _is_connected_to_wide_transverse_groove(
                    segment,
                    wide_clusters_by_column,
                    min_connected_length_px=min_segment_length_px,
                ):
                    continue
                if not _has_clear_sipe_sides(binary, segment):
                    continue

                center_y, mean_width, first_column, last_column = accepted_segment
                _paint_segment_mask(line_mask, segment)
                raw_segments.append((center_y, mean_width, first_column, last_column))

    deduped_segments = _dedupe_segments(raw_segments, dedup_distance_px)
    positions = [position for position, _width, _first_column, _last_column in deduped_segments]
    widths = [width for _position, width, _first_column, _last_column in deduped_segments]
    return positions, len(deduped_segments), line_mask, widths


def _split_rows_into_clusters(component_rows: np.ndarray, top_offset: int) -> list[tuple[int, int]]:
    row_clusters: list[tuple[int, int]] = []
    cluster_start = int(component_rows[0])
    for row_index in range(1, len(component_rows)):
        if int(component_rows[row_index]) - int(component_rows[row_index - 1]) > 2:
            row_clusters.append((cluster_start + top_offset, int(component_rows[row_index - 1]) + top_offset))
            cluster_start = int(component_rows[row_index])
    row_clusters.append((cluster_start + top_offset, int(component_rows[-1]) + top_offset))
    return row_clusters


def _collect_wide_clusters_by_column(binary: np.ndarray, min_width_px: int) -> WideClusterMap:
    image_height, image_width = binary.shape
    max_width_px = image_height // 2
    wide_clusters_by_column: WideClusterMap = {}

    for column_index in range(image_width):
        dark_rows = np.where(binary[:, column_index] > 0)[0]
        if len(dark_rows) == 0:
            continue

        wide_clusters = [
            (start_row, end_row)
            for start_row, end_row in _split_rows_into_clusters(dark_rows, 0)
            if min_width_px <= (end_row - start_row + 1) <= max_width_px
        ]
        if wide_clusters:
            wide_clusters_by_column[column_index] = wide_clusters

    return wide_clusters_by_column


def _validate_segment(
    segment: TrackData,
    min_width_px: int,
    max_width_px: int,
    min_segment_length_px: int,
) -> Segment | None:
    if not segment:
        return None

    first_column = segment[0][0]
    last_column = segment[-1][0]
    segment_width = last_column - first_column + 1
    if segment_width < min_segment_length_px:
        return None

    mean_width = float(np.mean([column_width for _column_index, _center_y, column_width in segment]))
    if mean_width < min_width_px or mean_width > max_width_px:
        return None

    center_y = float(np.mean([center_y for _column_index, center_y, _column_width in segment]))
    return center_y, mean_width, first_column, last_column


def _is_connected_to_wide_transverse_groove(
    segment: TrackData,
    wide_clusters_by_column: WideClusterMap,
    min_connected_length_px: int,
    max_initial_gap_columns: int = 5,
    max_scan_columns: int = 40,
) -> bool:
    if not segment or not wide_clusters_by_column:
        return False

    first_column, first_center_y, _first_width = segment[0]
    last_column, last_center_y, _last_width = segment[-1]
    segment_length_px = last_column - first_column + 1
    required_connected_columns = min(min_connected_length_px, max(8, segment_length_px // 2))

    slope = (last_center_y - first_center_y) / max(1, last_column - first_column)
    endpoints = ((segment[0], -1), (segment[-1], 1))

    for (column_index, center_y, column_width), direction in endpoints:
        vertical_padding = max(2, int(round(column_width)))
        matching_columns = 0
        gap_columns = 0
        has_seen_wide_cluster = False

        for distance in range(1, max_scan_columns + 1):
            neighbor_column = column_index + direction * distance
            expected_center_y = center_y + slope * (neighbor_column - column_index)
            has_matching_wide_cluster = False

            for start_row, end_row in wide_clusters_by_column.get(neighbor_column, []):
                if start_row - vertical_padding <= expected_center_y <= end_row + vertical_padding:
                    has_matching_wide_cluster = True
                    break

            if has_matching_wide_cluster:
                matching_columns += 1
                has_seen_wide_cluster = True
                gap_columns = 0
                if matching_columns >= required_connected_columns:
                    return True
                continue

            if not has_seen_wide_cluster:
                gap_columns += 1
                if gap_columns > max_initial_gap_columns:
                    break

    return False


def _has_clear_sipe_sides(binary: np.ndarray, segment: TrackData) -> bool:
    if not segment:
        return False

    above_values: list[int] = []
    below_values: list[int] = []
    for column_index, center_y, column_width in segment:
        side_offset = max(4, int(round(column_width)) + 2)
        center_row = int(round(center_y))
        above_row = center_row - side_offset
        below_row = center_row + side_offset
        if 0 <= above_row < binary.shape[0]:
            above_values.append(int(binary[above_row, column_index] > 0))
        if 0 <= below_row < binary.shape[0]:
            below_values.append(int(binary[below_row, column_index] > 0))

    if not above_values or not below_values:
        return False

    above_dark_ratio = float(np.mean(above_values))
    below_dark_ratio = float(np.mean(below_values))
    return above_dark_ratio <= 0.25 and below_dark_ratio <= 0.25


def _paint_segment_mask(line_mask: np.ndarray, segment: TrackData) -> None:
    for column_index, center_y, column_width in segment:
        start_row = max(0, int(round(center_y - column_width / 2.0)))
        end_row = min(line_mask.shape[0] - 1, int(round(center_y + column_width / 2.0)))
        line_mask[start_row: end_row + 1, column_index] = 255


def _dedupe_segments(raw_segments: list[Segment], dedup_distance_px: float) -> list[Segment]:
    raw_segments.sort(key=lambda item: item[0])
    deduped_segments: list[Segment] = []

    for center_y, width, first_column, last_column in raw_segments:
        merged = False
        for segment_index, (existing_center_y, existing_width, existing_first_column, existing_last_column) in enumerate(deduped_segments):
            if abs(center_y - existing_center_y) >= dedup_distance_px:
                continue

            overlap = max(0, min(last_column, existing_last_column) - max(first_column, existing_first_column) + 1)
            min_span = min(last_column - first_column + 1, existing_last_column - existing_first_column + 1)
            if min_span > 0 and overlap / min_span > 0.5:
                deduped_segments[segment_index] = (
                    (existing_center_y + center_y) / 2.0,
                    max(existing_width, width),
                    min(existing_first_column, first_column),
                    max(existing_last_column, last_column),
                )
                merged = True
                break

        if not merged:
            deduped_segments.append((center_y, width, first_column, last_column))

    return deduped_segments


def _draw_debug_image(
    image: np.ndarray,
    line_mask: np.ndarray,
    count: int,
) -> np.ndarray:
    debug_image = image.copy()
    overlay = np.zeros_like(debug_image)
    overlay[line_mask > 0] = (200, 100, 0)
    debug_image = cv2.addWeighted(debug_image, 0.7, overlay, 0.3, 0)

    contours, _hierarchy = cv2.findContours(
        (line_mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_height, image_width = debug_image.shape[:2]
    for contour in sorted(contours, key=lambda item: cv2.boundingRect(item)[:2]):
        left, top, width, height = cv2.boundingRect(contour)
        pad = 2
        box_left = max(0, left - pad)
        box_top = max(0, top - pad)
        box_right = min(image_width - 1, left + width + pad)
        box_bottom = min(image_height - 1, top + height + pad)
        cv2.rectangle(debug_image, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_thickness = 1
    text_color = (255, 255, 255)
    background_color = (0, 0, 0)
    labels = [f"sipes:{count}"]
    text_y = 10
    for label in labels:
        (text_width, text_height), _baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(debug_image, (1, text_y - text_height - 1), (3 + text_width, text_y + 2), background_color, -1)
        cv2.putText(debug_image, label, (2, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        text_y += text_height + 4

    return debug_image


__all__ = ["detect_transverse_sipes"]
