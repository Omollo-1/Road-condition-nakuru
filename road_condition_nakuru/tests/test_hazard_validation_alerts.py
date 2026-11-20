import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from scripts.hazard_validation_alerts import (
    aggregate_detections_by_segment,
    aggregate_images_by_segment,
    classify_segment_condition,
    condition_confusion_matrix,
    create_stratified_ground_truth_sample,
    deduplicate_detections,
)


class HazardValidationTests(unittest.TestCase):
    def test_deduplication_counts_unique_instances_and_independent_sequences(self):
        detections = pd.DataFrame(
            [
                {
                    "image_id": "image-1",
                    "sequence_id": "sequence-a",
                    "hazard_type": "pothole",
                    "confidence": 0.8,
                    "bbox_area_px": 100,
                    "lat": -0.30,
                    "lon": 36.06,
                    "compass_angle": 90,
                    "capture_date": "2026-03-01",
                    "osm_segment_id": "segment-1",
                },
                {
                    "image_id": "image-2",
                    "sequence_id": "sequence-a",
                    "hazard_type": "pothole",
                    "confidence": 0.9,
                    "bbox_area_px": 120,
                    "lat": -0.300001,
                    "lon": 36.060001,
                    "compass_angle": 95,
                    "capture_date": "2026-03-02",
                    "osm_segment_id": "segment-1",
                },
                {
                    "image_id": "image-3",
                    "sequence_id": "sequence-b",
                    "hazard_type": "pothole",
                    "confidence": 0.7,
                    "bbox_area_px": 90,
                    "lat": -0.3002,
                    "lon": 36.06,
                    "compass_angle": 90,
                    "capture_date": "2026-03-03",
                    "osm_segment_id": "segment-1",
                },
            ]
        )

        unique = deduplicate_detections(detections)
        aggregated = aggregate_detections_by_segment(detections)

        self.assertEqual(len(unique), 2)
        self.assertEqual(int(aggregated.loc[0, "n_unique_defects"]), 2)
        self.assertEqual(int(aggregated.loc[0, "n_corroborating_captures"]), 2)

    def test_different_camera_headings_do_not_merge_same_location_defects(self):
        detections = pd.DataFrame(
            [
                {
                    "image_id": "forward",
                    "hazard_type": "crack",
                    "confidence": 0.8,
                    "bbox_area_px": 100,
                    "lat": -0.30,
                    "lon": 36.06,
                    "compass_angle": 0,
                    "capture_date": "2026-03-01",
                    "osm_segment_id": "segment-1",
                },
                {
                    "image_id": "reverse",
                    "hazard_type": "crack",
                    "confidence": 0.8,
                    "bbox_area_px": 100,
                    "lat": -0.30,
                    "lon": 36.06,
                    "compass_angle": 180,
                    "capture_date": "2026-03-02",
                    "osm_segment_id": "segment-1",
                },
            ]
        )

        self.assertEqual(len(deduplicate_detections(detections)), 2)

    def test_segments_without_defect_evidence_are_unassessed_with_coverage(self):
        edges = gpd.GeoDataFrame(
            {"osm_segment_id": ["segment-1"], "name": ["Main Road"]},
            geometry=[LineString([(36.06, -0.30), (36.061, -0.30)])],
            crs="EPSG:4326",
        )
        images = pd.DataFrame(
            [
                {
                    "osm_segment_id": "segment-1",
                    "image_id": "image-1",
                    "thumb_url": "https://example.test/image.jpg",
                    "capture_date": "2026-03-10",
                },
                {
                    "osm_segment_id": "segment-1",
                    "image_id": "image-2",
                    "thumb_url": "https://example.test/image-2.jpg",
                    "capture_date": "2026-03-11",
                },
            ]
        )
        alerts = gpd.GeoDataFrame(
            columns=[
                "osm_segment_id",
                "hazard_type",
                "severity_score",
                "severity_label",
                "n_unique_defects",
                "n_corroborating_images",
                "last_seen",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        coverage = aggregate_images_by_segment(images)
        result = classify_segment_condition(alerts, edges, coverage)

        self.assertEqual(result.loc[0, "condition_label"], "Unassessed")
        self.assertEqual(int(result.loc[0, "image_count"]), 2)
        self.assertEqual(
            pd.Timestamp(result.loc[0, "last_capture_date"]),
            pd.Timestamp("2026-03-11"),
        )

    def test_ground_truth_sample_is_reproducible_and_stratified(self):
        rows = []
        for highway_class in ("primary", "residential"):
            for condition in ("Good", "Moderate", "Poor", "Unassessed"):
                for index in range(20):
                    rows.append(
                        {
                            "osm_segment_id": f"{highway_class}-{condition}-{index}",
                            "road_name": "Test road",
                            "highway_class": highway_class,
                            "condition_label": condition,
                        }
                    )
        roads = gpd.GeoDataFrame(rows, geometry=[None] * len(rows), crs="EPSG:4326")

        first = create_stratified_ground_truth_sample(roads)
        second = create_stratified_ground_truth_sample(roads)

        self.assertEqual(len(first), 120)
        self.assertEqual(
            first["osm_segment_id"].tolist(), second["osm_segment_id"].tolist()
        )
        self.assertEqual(first["observed_condition"].tolist(), [""] * 120)
        self.assertEqual(
            set(zip(first["highway_class"], first["model_condition"])),
            set(zip(roads["highway_class"], roads["condition_label"])),
        )

    def test_confusion_matrix_keeps_unassessed_predictions_visible(self):
        matrix = condition_confusion_matrix(
            pd.Series(["Poor", "Good"]),
            pd.Series(["Unassessed", "Good"]),
        )

        self.assertEqual(matrix.loc["Poor", "Unassessed"], 1)
        self.assertEqual(matrix.loc["Good", "Good"], 1)
        self.assertEqual(int(matrix.to_numpy().sum()), 2)


if __name__ == "__main__":
    unittest.main()
