# Deep-learning-final-project
CNN Prediction of Neighborhood Walkability from Satellite Imagery


## Project Overview and Purpose

This project investigates predicting neighborhood walkability using overhead satellite imagery. "Walkability" is defined as the degree to which a built environment supports walking as a main mode of transportation, based on factors like sidewalk and road infrastructure, building and intersection density, etc. Walkability in urban environments plays a large role in things like public health, sustainability, and equitable city design, and is interesting to me personally as a spatial data science student.

The specific aim of this project is to assess whether a convolutional neural network trained solely on overhead aerial imagery can accurately classify neighborhood walkability scores without any ground-level data (i.e. Google Streetview).


## Dataset

The dataset for this project was constructed from scratch, consisting of ~8,400 satellite image patches across the United States, labeled by neighborhood walkability class (low / medium / high) These data samples were derived from the [EPA National Walkability Index (NWI) geodatabase](https://www.epa.gov/smartgrowth/smart-location-mapping#walkability) and [Mapbox Static Images API satellite imagery](https://docs.mapbox.com/api/maps/static-images/). Each image is a 512×512 JPEG aerial photo (later normalized to 224x224) centered on a US census block group centroid, fetched at zoom level 16 (~0.3–0.6m resolution).

The dataset was built by filtering the NWI dataset to a representative subset of cities & fetching satellite imagery for the centroid coordinates of each neighborhood, effectively creating a large dataset of satellite imagery patches and their corresponding numerically encoded labels (low=0, medium=1, high=2).

See [data.md](https://github.com/siegelhannah/deeplearning-walkability-cnn/blob/main/walkability/dataset/data.md) for more detailed dataset description, and [dataset_download.py](https://github.com/siegelhannah/deeplearning-walkability-cnn/blob/main/walkability/dataset/dataset_download.py) for specific construction details.

The dataset is stored on Talapas at `/projects/dsci410_510/data/walkability_dataset/`.


## Models

The models trained on this dataset include:
1. A CNN built from scratch
2. A pre-trained, fine-tuned ResNet18 model (pre-trained on ImageNet data)

The from-scratch CNN has two convolutional blocks (with conv layers, batch normalization, and max pooling) and 0.2 dropout on the fc linear layers. It uses a learning rate of 1e-5 and Early Stopping patience=15.

The pre-trained ResNet18 has 0.3 dropout on the fine-tuned fc linear layers, a learning rate of 1e-5, and Early Stopping patience=10.


## Training Instructions






## Results
In progress

## Discussion and Limitations


## General Usage:

1. Clone the repo and install:

```bash
git clone https://github.com/siegelhannah/deeplearning-walkability-cnn.git
cd deeplearning-walkability-cnn
pip install -e .
```

2. Open notebooks/data_demo.ipynb

3. The dataset located at / can be loaded from:
/projects/dsci410_510/data/walkability_dataset/

4. Run all cells.



