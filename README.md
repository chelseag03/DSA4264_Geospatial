# Estimating the Effect of “Good” School Proximity on HDB Resale Prices

HDB resale price analysis based on school proximity effects.

## Technical Report
The full report is available [here](https://chelseag03.github.io/DSA4264_Grp7_Geospatial/).

## Files to Run
Please run the scripts in the following order:

1. `good_pri_sch_data.py`  
   Scrapes, cleans, and identifies good primary schools.  
   Main output:
   - `good_primary_schools.csv`

2. `hdb_amenity_data_cleaning.py`  
   Cleans and prepares the HDB resale and amenity datasets.  
   Main output:
   - `hdb_with_amenities.csv`

3. `dist_bands.py`  
   Uses the outputs from the previous two scripts to create school-distance band features for HDB blocks.  
   Main output:
   - `hdb_with_school_features.csv`

4. `model_building.py`  
   Runs the final modelling pipeline using the engineered dataset from `dist_bands.py`.
   Builds and evaluates OLS (hedonic pricing), Ridge, Lasso, Random Forest, and XGBoost models to estimate the effect of good primary school proximity on HDB resale prices.
   Main output:
   - `model_comparison.png`
   - `school_effect_ols.png`
   - `actual_vs_predicted_rf.png`
   - `feature_importance_rf.png`

## Required Input Files

Before running the scripts, download and place these raw files in the working directory.

### For `good_pri_sch_data.py`
- `Generalinformationofschools.csv` from [data.gov.sg](https://data.gov.sg/datasets/d_688b934f82c1059ed0a6993d2a829089/view?dataExplorerPage=7)

### For `hdb_amenity_data_cleaning.py`
- HDB resale transaction files (5 datasets) from [data.gov.sg HDB resale collection](https://data.gov.sg/collections/189/view):
  - `Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv`
  - `Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv`
  - `Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv`
  - `Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv`
  - `Resale flat prices based on registration date from Jan-2017 onwards.csv`
- `HDBPropertyInformation.csv` from [data.gov.sg](https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view)
- `road_name_road_code_jan2024.xlsx` from [LTA GIS Data Hub](https://www.lta.gov.sg/content/dam/ltagov/industry_innovations/industry_matters/development_construction_resources/Street_Work_Proposals/Standards_and_Specifications/GIS_Data_Hub/road_name_road_code_jan2024.xlsx)
- MRT/LRT exit data from [data.gov.sg MRT/LRT collection](https://data.gov.sg/collections/367/view)
- `mrt_lrt_stations_2025-01-14.csv` from [Kaggle](https://www.kaggle.com/datasets/lzytim/full-list-of-mrt-and-lrt-stations-in-singapore?select=mrt_lrt_stations_2025-01-14.csv)
- Hawker centre data from [data.gov.sg](https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view)
- `HDBExistingBuilding.geojson` from [data.gov.sg](https://data.gov.sg/datasets/d_16b157c52ed637edd6ba1232e026258d/view)
  
### For `dist_bands.py`
- `SLACadastralLandParcel.geojson` from [data.gov.sg](https://data.gov.sg/datasets/d_e7395d743076a2bcc487b0d12b9bf33b/view)

## Intermediate Files Generated
Running the scripts in order will generate these intermediate files:
- `good_primary_schools.csv`
- `hdb_with_amenities.csv`

## Final Output
- `hdb_with_school_features.csv`

## How to Run
Run each script one at a time:

```bash
python good_pri_sch_data.py
python hdb_amenity_data_cleaning.py
python dist_bands.py
python model_building.py
