# Walkability Dataset

## Overview
~8,400 satellite image patches labeled by neighborhood walkability class (low / medium / high),
derived from the EPA National Walkability Index and Mapbox satellite imagery.
Each image is a 512×512 JPEG aerial photo centered on a US census block group centroid,
fetched at zoom level 16 (~0.3–0.6m resolution).

## Labels
Walkability scores (1–20) from the EPA NWI are binned into 3 classes using national
quantile cuts across the selected cities:
- **low** → 0
- **medium** → 1  
- **high** → 2

~2,800 samples per class (balanced).

## Cities
Top 19 US cities by population, selected to cover a wide range of walkability profiles.
Capped at 150 samples per class per city to ensure balanced representation.

## Dataset Location (Talapas)
The full dataset is stored on Talapas at: /projects/dsci410_510/data/walkability_dataset/

This directory contains:
- `low/`, `medium/`, `high/` — image folders
- `data.csv` — full metadata manifest
- `train.csv`, `develop.csv`, `test.csv` — pre-split subsets (60/20/20)

## How to Load
```python
from walkability.dataset.dataloader import get_data_loaders

train_loader, val_loader, test_loader = get_data_loaders(
    base_path="/projects/dsci410_510/data/walkability_dataset/",
    name="all"
)
```

## How the Dataset Was Built
See `dataset_building.py` in this folder. It requires:
- The EPA NWI geodatabase (`Natl_WI.gdb`), downloadable from:
  https://www.epa.gov/smartgrowth/smart-location-mapping#walkability
- A Mapbox API key (free tier at https://account.mapbox.com)

Running `dataset_building.py` will fetch all images and generate the CSV splits automatically.
