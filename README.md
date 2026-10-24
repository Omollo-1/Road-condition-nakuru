# Nakuru Road Condition Assessment

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Omollo-1/Road-condition-nakuru/blob/main/road_condition_nakuru/models/Nakuru_Road_Condition_Assessment.ipynb)

A notebook-based pilot for mapping road-surface hazards in Nakuru City. It
fine-tunes a YOLO detector on a small RDD2022 subset, retrieves street-level
imagery from Mapillary when configured, associates detections with the road
network, and exports maps and analysis tables.

The same notebook can be run from a local Python environment, Jupyter, or
Google Colab. GPU acceleration is recommended for model training.

## Final project status

This repository contains the Nakuru road-condition dashboard and monitoring
workflow in a validated, repo-ready state. The deliverable at
`road_condition_nakuru/dashboard.html` renders a GIS-style overview of the
Nakuru road network, includes the road opacity control, shows good/moderate/
poor/unknown status overlays, and presents hazard data with defect metadata such
as missing signage, missing road lights, and missing guardrails. The dashboard
has been checked in the browser and the main KPI cards, map panel, filter
controls, and defect queue are all present and operational.

## Repository layout

- [`road_condition_nakuru/models/Nakuru_Road_Condition_Assessment.ipynb`](road_condition_nakuru/models/Nakuru_Road_Condition_Assessment.ipynb) —
  main end-to-end analysis notebook.
- [`road_condition_nakuru/scripts/download_rdd2022_subset.py`](road_condition_nakuru/scripts/download_rdd2022_subset.py) —
  resumable downloader for a capped RDD2022 training/validation subset.
- [`road_condition_nakuru/requirements.txt`](road_condition_nakuru/requirements.txt) —
  Python dependencies.
- [`road_condition_nakuru/README.md`](road_condition_nakuru/README.md) —
  notebook details and output-file reference.

## Local Python and Jupyter

The following commands use PowerShell on Windows. Install Python 3.10 or newer,
then run from a terminal:

```powershell
git clone https://github.com/Omollo-1/Road-condition-nakuru.git
cd Road-condition-nakuru\road_condition_nakuru
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\download_rdd2022_subset.py
jupyter lab models\Nakuru_Road_Condition_Assessment.ipynb
```

On macOS/Linux, activate the environment with
`source .venv/bin/activate` and use forward-slash paths. Run the downloader from
the `road_condition_nakuru` folder; it selects 2,500 training and 500 validation
images by default. The download is resumable and requires internet access and
about 1 GB of free disk space.

Run the notebook cells from top to bottom. Its quick-training preset uses five
epochs, 512-pixel images, and 5% of the training subset to reduce run time. It
caches the fine-tuned weights locally so subsequent runs can skip training.
Increase the training settings in the notebook for a more complete experiment.

## Jupyter in an existing environment

From the `road_condition_nakuru` folder, install the dependencies, download the
dataset, and launch Jupyter:

```bash
python -m pip install -r requirements.txt
python scripts/download_rdd2022_subset.py
jupyter lab models/Nakuru_Road_Condition_Assessment.ipynb
```

The notebook discovers the project directory from its current working
directory. If using an alternative YOLO dataset, set `RDD2022_DATASET_DIR` to
its location before launching Jupyter.

## Google Colab

Open the notebook with the badge above, choose a GPU runtime (**Runtime →
Change runtime type → T4 GPU**, if available), then run this setup cell before
the notebook's first code cell:

```python
!git clone --depth 1 https://github.com/Omollo-1/Road-condition-nakuru.git /content/Road-condition-nakuru
%cd /content/Road-condition-nakuru/road_condition_nakuru
%pip install -r requirements.txt
!python scripts/download_rdd2022_subset.py
```

The dataset download is about 876 MB and Colab storage is temporary. Re-run the
setup after a runtime reset. Then run the notebook from top to bottom.

## Mapillary and other credentials

Mapillary imagery is optional. Without a token, the notebook skips image
retrieval and reports roads without observations as **Unknown**; it does not
label unsurveyed roads as good.

- **Windows/PowerShell:** set the token in the same terminal used to launch
  Jupyter. `Read-Host` avoids putting the token in the command itself:

  ```powershell
  $env:MAPILLARY_ACCESS_TOKEN = Read-Host "Mapillary access token"
  jupyter lab models\Nakuru_Road_Condition_Assessment.ipynb
  ```

- **macOS/Linux:** export it in the terminal before launching Jupyter:

  ```bash
  read -s MAPILLARY_ACCESS_TOKEN
  export MAPILLARY_ACCESS_TOKEN
  jupyter lab models/Nakuru_Road_Condition_Assessment.ipynb
  ```

- **Google Colab:** add a secret named `MAPILLARY_ACCESS_TOKEN` in the
  notebook's **Secrets** panel and grant the notebook access. The notebook reads
  that secret without saving it in the notebook.

Never paste access tokens into notebook source, commit them, or share them in
screenshots. **Any token previously shared in chat should be revoked and
replaced** before use. The notebook can also optionally use Earth Engine; set
up Google Earth Engine authentication and configure `GEE_PROJECT_ID` if that
layer is needed.

## Analysis outputs

Generated artifacts are written under `road_condition_nakuru/outputs/`:

- Road-condition GeoJSON and CSV summaries.
- Hazard detections and alerts in GeoJSON/CSV.
- Interactive road-condition and street-view HTML maps.
- Model metrics, run summary, and optional ground-truth validation metrics.
- Ultralytics training curves, evaluation plots, and model weights under
  `outputs/training/`.

Outputs depend on available road data and imagery; Mapillary-derived results
require a valid token and image coverage in the pilot area. Ground-truth
precision/recall is only calculated when a completed survey file is supplied.
See the notebook and the subproject README for exact filenames and validation
details.

## Dataset and attribution

RDD2022 is downloaded on demand from the public
[dronefreak/RDD2022 Hugging Face dataset](https://huggingface.co/datasets/dronefreak/RDD2022),
an unofficial YOLO-format re-export. Its source repository states the data is
available under CC BY-SA 4.0; check the source's current terms and attribution
requirements before redistributing. The downloader maps the three crack
categories to `crack` and pothole to `pothole`. The complete Kaggle archive is
not downloaded.

Images, labels, model weights, caches, and generated analysis outputs are
local-only and excluded from Git. The downloader creates a `SOURCE.txt`
attribution note and `dataset.yaml` beside the downloaded data.

## Limitations

This is a quick pilot configuration, not a full-scale or independently
validated road survey. Detection quality depends on the training subset,
Mapillary coverage, image conditions, and local validation data. Review
predictions and compare them with field observations before using results for
operational or funding decisions.
