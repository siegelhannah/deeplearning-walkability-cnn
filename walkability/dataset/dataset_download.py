'''
building dataset for deep learning 410 project

Data sources:
EPA NWI: https://www.epa.gov/smartgrowth/smart-location-mapping#walkability  
Mapbox Static Images API: https://docs.mapbox.com/api/maps/static-images/ 
'''

import os
import time
import requests
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import fiona
from sklearn.model_selection import train_test_split



# Cities to include, by CBSA_Name: Largest-population cities from each class
# 3 classes determined by average city walkability: low, medium, high

# Top 21 U.S. cities by population, span a wide walkability range b/c of size (i.e. include city, suburbs, etc.)
# CBSA = "city area", aka counties anchored by a central urban core (at least 10,000 residents), 
# plus adjacent counties that have strong economic and commuting ties to that core.
TARGET_CITIES = [
    "New York-Newark-Jersey City, NY-NJ-PA",
    "Los Angeles-Long Beach-Anaheim, CA",
    "Chicago-Naperville-Elgin, IL-IN-WI", 
    "Houston-The Woodlands-Sugar Land, TX",       # Texas
    "Phoenix-Mesa-Chandler, AZ", 
    "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD", 
    # "San Antonio-New Braunfels, TX",              # Texas
    "San Diego-Chula Vista-Carlsbad, CA",
    # "Dallas-Fort Worth-Arlington, TX",            # Texas
    "Jacksonville, FL",
    "San Jose-Sunnyvale-Santa Clara, CA", 
    "Austin-Round Rock-Georgetown, TX",           # Texas
    "Charlotte-Concord-Gastonia, NC-SC",
    "Columbus, OH", 
    "Indianapolis-Carmel-Anderson, IN", 
    "San Francisco-Oakland-Berkeley, CA",
    "Seattle-Tacoma-Bellevue, WA",
    "Denver-Aurora-Lakewood, CO",
    "Nashville-Davidson--Murfreesboro--Franklin, TN",
    "Oklahoma City, OK",
    "Washington-Arlington-Alexandria, DC-VA-MD-WV"
]
# cut San Antonio and Dallas b/c of skew towards Texas cities - now top 19 cities


# KEYS / PATHS / SETTINGS FOR RUNNING SCRIPT:

MAPBOX_API_KEY = "sk.eyJ1IjoiaGFubmFoc2llZ2VsIiwiYSI6ImNtcGViMTBtZTAyZDMyd29xcnZra2Fwb2QifQ.TqLam3Ip6YSq7mD-wqTPkg"
NWI_GDB_PATH = "Natl_WI.gdb"
OUTPUT_DIR = "walkability_dataset"
IMAGE_SIZE = 512 # pixels (square)
ZOOM_LEVEL = 16 # 16 = ~0.3-0.6m resolution for US cities
SLEEP_BETWEEN = 0.001 # sec between API calls (1250/min mapbox limit)


def load_nwi(gdb_path, cities):
    """
    # 1: LOAD AND FILTER THE NATIONAL WALKABILITY INDEX GDF
    """
    # select specific layer name
    df = gpd.read_file(gdb_path, layer="NationalWalkabilityIndex")

    # Filter to target cities
    df = df[df["CBSA_Name"].isin(cities)].copy()

    df = df.to_crs("EPSG:4326")  # reproject to degrees (to match with MapBox API coords)

    # get centroids of each block group
    df["centroid_lon"] = df.geometry.centroid.x
    df["centroid_lat"] = df.geometry.centroid.y

    df = pd.DataFrame(df.drop(columns="geometry"))  # drop geometry for speed

    # drop any missing centroid coords or walkability scores
    df = df.dropna(subset=["centroid_lat", "centroid_lon", "NatWalkInd"])

    # Bin walkability scores by quantiles (3 classes) to have balanced class sizes for training
    df["label"] = pd.qcut(df["NatWalkInd"], q=3, labels=["low", "medium", "high"])
    df = df.dropna(subset=["label"]) # drop nans

    # Limit number of block groups per city (to balance samples per city)
    # also cap to 150 samples per class, per city, for balanced training data
    sampled = []
    for city in df["CBSA"].unique():
        for lbl in ["low", "medium", "high"]:
            subset = df[(df["CBSA"] == city) & (df["label"] == lbl)]
            sampled.append(subset.sample(min(len(subset), 150), random_state=42))

    df = pd.concat(sampled).reset_index(drop=True)
    
    return df[["GEOID10", "centroid_lat", "centroid_lon", "NatWalkInd", "label", "CBSA_Name"]]


def fetch_image(lat, lon, api_key, zoom, size):
    """
    # 2: Download satellite imagery from Mapbox Static Images API for each centroid
    API docs: https://docs.mapbox.com/api/maps/static-images/
    """
    # url w/ style specifications + MapBox watermark supression
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{size}x{size}?attribution=false&logo=false&access_token={api_key}"

    response = requests.get(url, timeout=10) # fetch image
    if response.status_code == 200:
        return response.content
    else:
        return None
    

def build_dataset(df, api_key, output_dir, zoom, size, sleep):
    """
    # 3. Pipeline to build the dataset (download/save all images, and build a metadata table about all samples)
    """
    # Create output folders for each class label (low, medium, high)
    for label in ["low", "medium", "high"]:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    records = [] # will become the dataset CSV (metadata + image urls per sample)
    failed  = [] # samples that failed to download

    for _, row in df.iterrows(): # go through NWI geodataframe rows (each block group)
        # select each column we want
        geoid = str(row["GEOID10"])
        lat = row["centroid_lat"]
        lon = row["centroid_lon"]
        label = str(row["label"])
        score = row["NatWalkInd"]
        city = row["CBSA_Name"]

        # set paths for downloading images
        img_filename = f"{geoid}.jpg"
        img_path = os.path.join(output_dir, label, img_filename)

        # Skip if already downloaded
        if os.path.exists(img_path):
            records.append({"geoid": geoid, "label": label, "score": score, "city": city, "path": img_path})
            continue

        # Fetch image
        img = fetch_image(lat, lon, api_key, zoom=zoom, size=size)

        if img:
            with open(img_path, "wb") as f: # write to path
                f.write(img)
            records.append({"geoid": geoid, "label": label, "score": score, "city": city, "path": img_path})
            # records = list of dicts (one dict per sample with all info/metadata)
        else:
            failed.append(geoid)
        # wait between downloads
        time.sleep(sleep)

    # Save dataset CSV (image paths + all other data)
    table = pd.DataFrame(records)
    table.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    print(f"\nDone. {len(records)} images saved, {len(failed)} failed.")
    print(f"Manifest saved to {output_dir}/data.csv")

    if failed:
        print(f"Failed GEOIDs: {failed}")

    return table



# TRAIN-TEST-SPLIT
def split_csv(data_csv, output_dir, train_size=0.60, val_size=0.20, random_state=42):
    """
    Split entire data.csv into train, dev/validation, and test CSVs and save to output_dir.
    ONLY RUN ONCE!
    
    Args:
        data_csv: Path to the full data.csv
        output_dir: Where to save train.csv, develop.csv, test.csv
        train_size: Fraction for training
        val_size: Fraction for validation (remainder goes to test)
        random_state (int): Random seed for reproducibility
    """
    df = pd.read_csv(data_csv)

    # split off test set first, then split remainder into train/val
    test_size = 1.0 - train_size - val_size
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), stratify=df["label"], random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=(test_size / (val_size + test_size)), stratify=temp_df["label"], random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "develop.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df)},  Val: {len(val_df)},  Test: {len(test_df)}")



if __name__ == '__main__':
    # RUN TO CREATE DATASET!
    gdf = load_nwi(NWI_GDB_PATH, TARGET_CITIES)
    print('loaded and filtered NWI gdf')
    build_dataset(gdf, MAPBOX_API_KEY, OUTPUT_DIR, ZOOM_LEVEL, IMAGE_SIZE, SLEEP_BETWEEN)
    split_csv(os.path.join(OUTPUT_DIR, "data.csv"), OUTPUT_DIR)  # split after building

