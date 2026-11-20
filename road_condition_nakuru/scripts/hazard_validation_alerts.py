"""Hazard aggregation, road-condition classification, and survey sampling."""

from __future__ import annotations

from math import hypot
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

HAZARD_BASE_WEIGHT = {
    "pothole": 0.6,
    "crack": 0.3,
    "poor_drainage": 0.5,
    "road_surface": 0.4,
    "road-signs": 0.5
}

ROAD_CLASS_MULTIPLIER = {
    "trunk": 1.4,
    "primary": 1.3,
    "secondary": 1.15,
    "tertiary": 1.0,
    "residential": 0.85,
    "unclassified": 0.8,
}

CONDITION_THRESHOLDS = {"Good": 0.35, "Moderate": 0.65}
CONDITION_LABELS = ("Good", "Moderate", "Poor", "Unassessed")
CONDITION_COLOR = {
    "Good": "#2A9D8F",
    "Moderate": "#F4A300",
    "Poor": "#E63946",
    "Unassessed": "#808080",
}


def bbox_area_score(bbox_area_px: float, image_area_px: float) -> float:
    """Return a normalized bounding-box-area proxy, capped at 25% of a frame."""
    if image_area_px <= 0:
        raise ValueError("image_area_px must be positive")
    frac = min(bbox_area_px / image_area_px, 0.25) / 0.25
    return round(frac, 3)


def classify_severity(
    hazard_type: str,
    confidence: float,
    bbox_area_px: float,
    image_area_px: float,
    road_class: str = "tertiary",
    n_corroborating_images: int = 1,
) -> Dict:
    """Combine the current pilot weights into a hazard severity score."""
    base = HAZARD_BASE_WEIGHT.get(hazard_type, 0.4)
    size = bbox_area_score(bbox_area_px, image_area_px)
    road_mult = ROAD_CLASS_MULTIPLIER.get(road_class, 1.0)
    corroboration = min((n_corroborating_images - 1) * 0.08, 0.2)
    score = (0.5 * base + 0.3 * size + 0.2 * confidence) * road_mult + corroboration
    score = round(min(score, 1.0), 3)

    if score >= 0.75:
        label = "critical"
    elif score >= 0.55:
        label = "high"
    elif score >= 0.35:
        label = "medium"
    else:
        label = "low"
    return {"severity_score": score, "severity_label": label}


def condition_from_score(score: float) -> str:
    """Map a measured defect score to a road-condition band."""
    if score < CONDITION_THRESHOLDS["Good"]:
        return "Good"
    if score < CONDITION_THRESHOLDS["Moderate"]:
        return "Moderate"
    return "Poor"


def _capture_id(row: pd.Series) -> str:
    sequence_id = row.get("sequence_id")
    if pd.notna(sequence_id) and str(sequence_id).strip():
        return str(sequence_id)
    image_id = row.get("image_id")
    return str(image_id) if pd.notna(image_id) else f"row-{row.name}"


def _heading_difference(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def deduplicate_detections(
    detections: pd.DataFrame,
    eps_m: float = 8.0,
    heading_tolerance_deg: float = 30.0,
) -> pd.DataFrame:
    """Cluster repeated detections of one defect by location and camera heading.

    Corroboration is counted later using unique Mapillary sequences when present,
    falling back to image IDs. Missing headings use location alone.
    """
    if eps_m <= 0:
        raise ValueError("eps_m must be positive")
    if not 0 < heading_tolerance_deg <= 180:
        raise ValueError("heading_tolerance_deg must be in (0, 180]")
    required = {
        "lat",
        "lon",
        "hazard_type",
        "image_id",
        "confidence",
        "bbox_area_px",
        "capture_date",
    }
    missing = required.difference(detections.columns)
    if missing:
        raise ValueError(f"Detections are missing required columns: {sorted(missing)}")
    output_columns = list(detections.columns) + [
        "instance_id",
        "capture_ids",
        "n_corroborating_images",
        "n_corroborating_captures",
    ]
    if detections.empty:
        return pd.DataFrame(columns=output_columns)

    valid = detections.dropna(subset=["lat", "lon"]).copy()
    invalid = detections.loc[detections[["lat", "lon"]].isna().any(axis=1)].copy()
    if "osm_segment_id" in valid:
        group_columns = ["osm_segment_id", "hazard_type"]
    else:
        group_columns = ["hazard_type"]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:21037", always_xy=True)
    records = []
    for group_key, group in valid.groupby(group_columns, dropna=False, sort=False):
        x_coords, y_coords = transformer.transform(
            group["lon"].astype(float).to_numpy(),
            group["lat"].astype(float).to_numpy(),
        )
        headings = pd.to_numeric(
            group.get("compass_angle", pd.Series(np.nan, index=group.index)),
            errors="coerce",
        ).to_numpy()
        coords = np.column_stack((x_coords, y_coords))
        distances = np.zeros((len(group), len(group)), dtype=float)
        heading_scale = eps_m / heading_tolerance_deg
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                spatial_distance = hypot(
                    coords[left, 0] - coords[right, 0],
                    coords[left, 1] - coords[right, 1],
                )
                if np.isfinite(headings[left]) and np.isfinite(headings[right]):
                    heading_distance = (
                        _heading_difference(headings[left], headings[right])
                        * heading_scale
                    )
                    distance = hypot(spatial_distance, heading_distance)
                else:
                    distance = spatial_distance
                distances[left, right] = distance
                distances[right, left] = distance

        labels = DBSCAN(eps=eps_m, min_samples=1, metric="precomputed").fit_predict(
            distances
        )
        for cluster_label in np.unique(labels):
            cluster = group.iloc[np.flatnonzero(labels == cluster_label)].copy()
            capture_ids = sorted({_capture_id(row) for _, row in cluster.iterrows()})
            record = cluster.iloc[0].to_dict()
            for column in ("confidence", "bbox_area_px"):
                if column in cluster:
                    values = pd.to_numeric(cluster[column], errors="coerce")
                    record[column] = values.mean() if column == "confidence" else values.max()
            record["lat"] = pd.to_numeric(cluster["lat"], errors="coerce").mean()
            record["lon"] = pd.to_numeric(cluster["lon"], errors="coerce").mean()
            if "capture_date" in cluster:
                dates = pd.to_datetime(cluster["capture_date"], errors="coerce")
                record["first_seen"] = dates.min()
                record["last_seen"] = dates.max()
            else:
                record["first_seen"] = pd.NaT
                record["last_seen"] = pd.NaT
            record["instance_id"] = f"{group_key!s}:{int(cluster_label)}"
            record["capture_ids"] = capture_ids
            record["n_corroborating_images"] = len(capture_ids)
            record["n_corroborating_captures"] = len(capture_ids)
            records.append(record)

    for row_index, row in invalid.iterrows():
        record = row.to_dict()
        capture_ids = [_capture_id(row)]
        record["instance_id"] = f"unlocated:{row_index}"
        record["capture_ids"] = capture_ids
        record["n_corroborating_images"] = 1
        record["n_corroborating_captures"] = 1
        records.append(record)
    return pd.DataFrame.from_records(records)


def _capture_count(capture_lists: pd.Series) -> int:
    return len({capture for captures in capture_lists for capture in captures})


def aggregate_detections_by_segment(
    detections: pd.DataFrame,
    segment_id_col: str = "osm_segment_id",
    hazard_type_col: str = "hazard_type",
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Deduplicate detections, then aggregate defect instances by segment/type."""
    if detections.empty:
        return pd.DataFrame(
            columns=[
                segment_id_col,
                hazard_type_col,
                "n_unique_defects",
                "n_corroborating_images",
                "n_corroborating_captures",
                "mean_confidence",
                "max_bbox_area_px",
                "mean_lat",
                "mean_lon",
                "first_seen",
                "last_seen",
            ]
        )
    unique = (
        deduplicate_detections(detections)
        if deduplicate or "capture_ids" not in detections
        else detections
    )
    unique = unique.dropna(subset=[segment_id_col, "lat", "lon"])
    return (
        unique.groupby([segment_id_col, hazard_type_col], dropna=False)
        .agg(
            n_unique_defects=("instance_id", "nunique"),
            n_corroborating_images=("capture_ids", _capture_count),
            n_corroborating_captures=("capture_ids", _capture_count),
            mean_confidence=("confidence", "mean"),
            max_bbox_area_px=("bbox_area_px", "max"),
            mean_lat=("lat", "mean"),
            mean_lon=("lon", "mean"),
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
        )
        .reset_index()
    )


def aggregate_images_by_segment(
    images_df: pd.DataFrame,
    segment_id_col: str = "osm_segment_id",
) -> pd.DataFrame:
    """Return per-segment image coverage and the most recent representative image."""
    columns = [
        segment_id_col,
        "representative_thumb_url",
        "representative_capture_date",
        "image_count",
        "last_capture_date",
    ]
    if images_df.empty:
        return pd.DataFrame(columns=columns)
    valid = images_df.dropna(subset=[segment_id_col]).copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)
    valid["capture_date"] = pd.to_datetime(valid["capture_date"], errors="coerce")
    summary = (
        valid.groupby(segment_id_col)
        .agg(
            image_count=("image_id", "nunique"),
            last_capture_date=("capture_date", "max"),
        )
        .reset_index()
    )
    latest = (
        valid.sort_values(["capture_date", "image_id"], na_position="first")
        .drop_duplicates(segment_id_col, keep="last")
        [[segment_id_col, "thumb_url", "capture_date"]]
        .rename(
            columns={
                "thumb_url": "representative_thumb_url",
                "capture_date": "representative_capture_date",
            }
        )
    )
    return summary.merge(latest, on=segment_id_col, how="left")[columns]


def generate_alerts(
    aggregated: pd.DataFrame,
    image_area_px: float,
    road_class_lookup: Optional[Dict[str, str]] = None,
) -> gpd.GeoDataFrame:
    """Create one spatial alert for each segment/hazard type."""
    road_class_lookup = road_class_lookup or {}
    records = []
    for _, row in aggregated.iterrows():
        road_class = road_class_lookup.get(row["osm_segment_id"], "tertiary")
        sev = classify_severity(
            hazard_type=row["hazard_type"],
            confidence=row["mean_confidence"],
            bbox_area_px=row["max_bbox_area_px"],
            image_area_px=image_area_px,
            road_class=road_class,
            n_corroborating_images=row["n_corroborating_images"],
        )
        records.append(
            {
                "osm_segment_id": row["osm_segment_id"],
                "hazard_type": row["hazard_type"],
                "severity_score": sev["severity_score"],
                "severity_label": sev["severity_label"],
                "n_unique_defects": row["n_unique_defects"],
                "n_corroborating_images": row["n_corroborating_images"],
                "n_corroborating_captures": row["n_corroborating_captures"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "geometry": Point(row["mean_lon"], row["mean_lat"]),
            }
        )
    columns = [
        "osm_segment_id",
        "hazard_type",
        "severity_score",
        "severity_label",
        "n_unique_defects",
        "n_corroborating_images",
        "n_corroborating_captures",
        "first_seen",
        "last_seen",
        "geometry",
    ]
    gdf = gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs="EPSG:4326")
    gdf["alert_color"] = gdf["severity_label"].map(
        {
            "critical": "#8B00FF",
            "high": "#E63946",
            "medium": "#F4A300",
            "low": "#2A9D8F",
        }
    )
    return gdf


def classify_segment_condition(
    alerts: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    representative_images: Optional[pd.DataFrame] = None,
    segment_id_col: str = "osm_segment_id",
    length_epsg: int = 21037,
) -> gpd.GeoDataFrame:
    """Roll alerts into segment condition without treating no evidence as Good."""
    if len(alerts):
        worst = (
            alerts.sort_values("severity_score", ascending=False)
            .drop_duplicates(segment_id_col)
            [[segment_id_col, "hazard_type", "severity_score"]]
            .rename(
                columns={
                    "hazard_type": "dominant_hazard_type",
                    "severity_score": "condition_score",
                }
            )
        )
        seg_stats = (
            alerts.groupby(segment_id_col)
            .agg(
                hazard_types_present=("hazard_type", lambda s: ", ".join(sorted(set(s)))),
                total_corroborating_images=("n_corroborating_images", "sum"),
                total_unique_defects=("n_unique_defects", "sum"),
                last_surveyed=("last_seen", "max"),
            )
            .reset_index()
        )
        seg_condition = worst.merge(seg_stats, on=segment_id_col)
    else:
        seg_condition = pd.DataFrame(
            columns=[
                segment_id_col,
                "dominant_hazard_type",
                "condition_score",
                "hazard_types_present",
                "total_corroborating_images",
                "total_unique_defects",
                "last_surveyed",
            ]
        )

    edges_condition = edges.merge(seg_condition, on=segment_id_col, how="left")
    if representative_images is not None:
        edges_condition = edges_condition.merge(
            representative_images, on=segment_id_col, how="left"
        )
    edges_condition["condition_score"] = pd.to_numeric(
        edges_condition["condition_score"], errors="coerce"
    )
    edges_condition["dominant_hazard_type"] = edges_condition[
        "dominant_hazard_type"
    ].fillna("none detected")
    edges_condition["hazard_types_present"] = edges_condition[
        "hazard_types_present"
    ].fillna("none detected")
    for column in ("total_corroborating_images", "total_unique_defects"):
        edges_condition[column] = (
            pd.to_numeric(edges_condition[column], errors="coerce").fillna(0).astype(int)
        )
    if "image_count" not in edges_condition:
        edges_condition["image_count"] = 0
    edges_condition["image_count"] = edges_condition["image_count"].fillna(0).astype(int)
    edges_condition["condition_label"] = edges_condition["condition_score"].map(
        lambda score: condition_from_score(score) if pd.notna(score) else "Unassessed"
    )
    edges_condition["condition_color"] = edges_condition["condition_label"].map(
        CONDITION_COLOR
    )
    edges_condition["length_m"] = (
        edges_condition.geometry.to_crs(epsg=length_epsg).length.round(1)
    )

    if "name" in edges_condition.columns:
        edges_condition["road_name"] = edges_condition["name"].apply(
            lambda name: name[0]
            if isinstance(name, list)
            else (name if pd.notna(name) else "Unnamed road")
        )
    return edges_condition


def condition_summary_table(edges_condition: gpd.GeoDataFrame) -> pd.DataFrame:
    """Summarize segment counts and road length by condition label."""
    return (
        edges_condition.groupby("condition_label")
        .agg(segment_count=("osm_segment_id", "count"), total_length_m=("length_m", "sum"))
        .reindex(CONDITION_LABELS, fill_value=0)
        .rename_axis("condition_label")
        .reset_index()
    )


def create_stratified_ground_truth_sample(
    edges_condition: gpd.GeoDataFrame,
    sample_size: int = 120,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select a reproducible 100-150 segment sample by class and model band."""
    if not 100 <= sample_size <= 150:
        raise ValueError("sample_size must be between 100 and 150")
    required = {"osm_segment_id", "highway_class", "condition_label"}
    missing = required.difference(edges_condition.columns)
    if missing:
        raise ValueError(f"Road segments are missing required columns: {sorted(missing)}")
    frame = edges_condition.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "osm_segment_id",
                "road_name",
                "highway_class",
                "model_condition",
                "observed_condition",
                "assessment_date",
                "assessor_id",
                "notes",
                "stratum_population",
                "sample_weight",
            ]
        )
    frame["highway_class"] = frame["highway_class"].fillna("unknown").astype(str)
    frame["condition_label"] = frame["condition_label"].fillna("Unassessed").astype(str)
    strata = (
        frame.groupby(["highway_class", "condition_label"], dropna=False)
        .size()
        .rename("population")
        .reset_index()
    )
    target_n = min(sample_size, len(frame))
    if len(strata) > target_n:
        raise ValueError("Sample size is too small to include every non-empty stratum")

    exact = strata["population"] * target_n / len(frame)
    allocations = np.floor(exact).astype(int).clip(upper=strata["population"])
    allocations = allocations.clip(lower=1)
    remainders = exact - np.floor(exact)
    while allocations.sum() > target_n:
        candidates = allocations[allocations > 1].index
        if len(candidates) == 0:
            raise ValueError("Unable to allocate the requested stratified sample")
        index = min(candidates, key=lambda item: (remainders.iloc[item], -allocations.iloc[item]))
        allocations.iloc[index] -= 1
    while allocations.sum() < target_n:
        candidates = allocations[allocations < strata["population"]].index
        if len(candidates) == 0:
            break
        index = max(
            candidates,
            key=lambda item: (exact.iloc[item] - allocations.iloc[item], -item),
        )
        allocations.iloc[index] += 1
    strata["sample_n"] = allocations

    selected = []
    for row in strata.itertuples(index=False):
        stratum = frame.loc[
            frame["highway_class"].eq(row.highway_class)
            & frame["condition_label"].eq(row.condition_label)
        ]
        chosen = stratum.sample(n=int(row.sample_n), random_state=random_state)
        chosen = chosen.copy()
        chosen["stratum_population"] = int(row.population)
        chosen["sample_weight"] = float(row.population / row.sample_n)
        selected.append(chosen)
    sample = pd.concat(selected, ignore_index=True).sample(
        frac=1, random_state=random_state
    )
    for column, default in (
        ("road_name", "Unnamed road"),
        ("assessment_date", ""),
        ("assessor_id", ""),
        ("notes", ""),
    ):
        if column not in sample:
            sample[column] = default
    sample["observed_condition"] = ""
    return sample[
        [
            "osm_segment_id",
            "road_name",
            "highway_class",
            "condition_label",
            "observed_condition",
            "assessment_date",
            "assessor_id",
            "notes",
            "stratum_population",
            "sample_weight",
        ]
    ].rename(columns={"condition_label": "model_condition"}).reset_index(drop=True)


def condition_confusion_matrix(
    observed: pd.Series,
    predicted: pd.Series,
    sample_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return an actual-row/predicted-column confusion matrix with fixed labels."""
    actual = observed.astype("string").str.strip().str.title()
    estimate = predicted.astype("string").str.strip().str.title()
    valid = actual.isin(CONDITION_LABELS) & estimate.isin(CONDITION_LABELS)
    if sample_weights is None:
        matrix = pd.crosstab(actual.loc[valid], estimate.loc[valid])
    else:
        weighted = pd.DataFrame(
            {
                "observed": actual.loc[valid],
                "predicted": estimate.loc[valid],
                "weight": sample_weights.loc[valid],
            }
        )
        matrix = weighted.pivot_table(
            index="observed",
            columns="predicted",
            values="weight",
            aggfunc="sum",
            fill_value=0,
        )
    return matrix.reindex(index=CONDITION_LABELS, columns=CONDITION_LABELS, fill_value=0)
