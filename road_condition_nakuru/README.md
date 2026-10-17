# Road Condition Assessment — Nakuru Pilot

A geospatial ML pipeline that pulls imagery for a Nakuru AOI, detects road damage
(potholes, cracks), geolocates and scores road segments, optionally compares
detections against a labeled ground-truth survey, and exports results for a road
authority. Built for GEOSK 2026.

## Folder structure

```
road_condition_nakuru/
├── README.md                    ← this file
├── requirements.txt              ← Python dependencies
├── models/
│   └── Nakuru_Road_Condition_Assessment.ipynb ← main notebook — run top to bottom
├── scripts/
│   └── download_rdd2022_subset.py ← capped, resumable RDD2022 subset downloader
├── data/                         ← ground-truth CSV and downloaded Mapillary images land here
│   └── ground_truth_template.csv ← copy this to ground_truth_nakuru.csv and fill in
└── outputs/                      ← exported GeoJSON/CSV results land here
```

## How to run

1. `pip install -r requirements.txt`
2. (Optional) Set `MAPILLARY_ACCESS_TOKEN` in the notebook environment to retrieve
   street-level images. Without it, the notebook still produces the analysis
   exports and marks unsurveyed roads as **Unknown**.
3. (Optional) Copy `data/ground_truth_template.csv` to
   `data/ground_truth_nakuru.csv`, replace the example rows with surveyed points,
   and run the notebook's export section to calculate precision/recall.
4. Download a capped RDD2022 pilot subset from the public Hugging Face mirror:
   `python scripts/download_rdd2022_subset.py`. It fetches 2,500 training and
   500 validation images individually, so it does not need the full 11.7 GB
   Kaggle archive. The source mirror is an unofficial YOLO re-export; see the
   generated `data/rdd2022_yolo_subset/SOURCE.txt` for attribution and licensing.
   The subset is resumable and the download tool remaps the source's four
   classes to this notebook's pothole/crack classes.
5. (Optional) Set `GEE_PROJECT_ID` if Earth Engine layers are needed.
6. Open `models/Nakuru_Road_Condition_Assessment.ipynb` and run cells top to
   bottom. The notebook uses the downloaded subset automatically; set
   `RDD2022_DATASET_DIR` to use a different local YOLO dataset.

## Faster model training

The default is a quick pilot configuration: 5 epochs, 512px images, and 5% of
the training set. The model is initialized from YOLOv8 nano, uses automatic
batch sizing on GPU, disables RAM image caching, and stops early if validation
does not improve. The resulting weights are cached at
`models/nakuru_rdd2022_best.pt`, so subsequent runs skip training. Increase
`EPOCHS` or `TRAINING_FRACTION` in notebook section 5c for a more thorough run.

## Notebook steps

| Step | What it does |
|---|---|
| 1 | Setup — installs, imports, config |
| 2 | Optional Earth Engine setup + Nakuru City AOI |
| 3 | Road network for the Nakuru city pilot area (OSMnx) |
| 4 | Street-level imagery pull (Mapillary) |
| 5 | Prepare RDD2022 and quick-fine-tune YOLO (cached on later runs) |
| 6 | Evaluate the detector and display training plots |
| 7 | Run inference over the pulled images |
| 8 | Geolocate detections onto the road network |
| 9 | Score road segments (condition class) |
| 10 | Visualize results on an interactive map |
| 11 | Optional ground-truth validation and export outputs |

GPU training is strongly recommended; CPU mode uses a smaller batch and no
data-loader workers. The notebook exports an explicit empty/unknown result when
there is no Mapillary imagery rather than treating unsurveyed roads as good.

## Outputs

The notebook writes these artifacts under `outputs/`:

- `nakuru_road_condition.geojson` — segment conditions
- `nakuru_road_alerts.geojson` and `nakuru_yolo_detections.csv` — hazard results
- `nakuru_condition_summary.csv` and `nakuru_severity_summary.csv` — summaries
- `nakuru_ground_truth_validation.json` — optional labeled-point precision,
  recall, and F1 (uses a configurable 25m location-match radius)
- `nakuru_model_metrics.json` and `nakuru_analysis_summary.json` — model and run
  metrics
- `nakuru_road_condition_map.html` and `nakuru_live_street_viewer.html` — maps
- `training/` — Ultralytics training curves, evaluation plots, and weights
