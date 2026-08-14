# Road Condition Assessment — Nakuru Pilot

A geospatial ML pipeline that pulls imagery for a Nakuru AOI, detects road damage
(potholes, cracks), geolocates and scores road segments, validates against ground
truth, and exports results for a road authority. Built for GEOSK 2026.

## Folder structure

```
road_condition_nakuru/
├── README.md                    ← this file
├── requirements.txt              ← Python dependencies
├── road_condition_pipeline.ipynb ← the main notebook — run this, top to bottom
├── models/                       ← put YOLOv8_Small_RDD.pt (or your fine-tuned best.pt) here
├── data/                         ← ground-truth CSV and downloaded Mapillary images land here
│   └── ground_truth_template.csv ← copy this to ground_truth_nakuru.csv and fill in
└── outputs/                      ← exported GeoJSON/CSV results land here
```

## How to run

1. `pip install -r requirements.txt`
2. Get a Google Earth Engine project: https://code.earthengine.google.com/
3. Get a Mapillary access token: https://www.mapillary.com/dashboard/developers
4. Download the pretrained detector into `models/`:
   ```
   curl -L -o models/YOLOv8_Small_RDD.pt \
     https://raw.githubusercontent.com/oracl4/RoadDamageDetection/main/models/YOLOv8_Small_RDD.pt
   ```
5. Open `road_condition_pipeline.ipynb` and fill in `GEE_PROJECT_ID` and
   `MAPILLARY_TOKEN` in Step 1. Run cells top to bottom.

## Notebook steps

| Step | What it does |
|---|---|
| 1 | Setup — installs, imports, config |
| 2 | Earth Engine auth + Nakuru County AOI + Sentinel-2 context layer |
| 3 | Road network for the Nakuru city pilot area (OSMnx) |
| 4 | Street-level imagery pull (Mapillary) |
| 5 | *(Optional)* Fine-tune the detector on your own labeled set — best in Colab |
| 6 | Load the detection model (fine-tuned or pretrained) |
| 7 | Run inference over the pulled images |
| 8 | Geolocate detections onto the road network |
| 9 | Score road segments (condition class) |
| 10 | Visualize results on an interactive map |
| 11 | Ground-truth validation (precision/recall) |
| 12 | Export outputs (GeoJSON + CSV) |

Step 5 is the only one that needs a GPU. Skip it entirely if you're fine using the
pretrained RDD2022 weights as-is — Step 6 falls back to downloading them
automatically.

## Notes

- Satellite imagery (used in Step 2) isn't fine-grained enough to detect individual
  potholes — that's what the street-level Mapillary pipeline (Steps 4, 7) is for.
  The satellite layer is for broader monitoring (vegetation encroachment, unpaved
  road erosion) at the county scale.
- Kenya's FAO GAUL *level 1* boundaries are the pre-2010 provinces, not the 47
  counties — Step 2 queries *level 2* instead, which is where "Nakuru" actually
  exists as a unit.
