# Estimating the Effect of “Good” School Proximity on HDB Resale Prices
HDB resale price analysis based on school proximity effects

## Technical Report
The full report is available here: https://chelseag03.github.io/DSA4264_Grp7_Geospatial/

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

4. MODEL script

## Required Input Files
Make sure the necessary raw data files are in the same working directory before running the scripts.

- HDB resale transaction files
- `HDBPropertyInformation.csv`
- `road_name_road_code_jan2024.xlsx`
- `LTAMRTStationExitGEOJSON.geojson`
- `mrt_lrt_stations_2025-01-14.csv`
- `HawkerCentresGEOJSON.geojson`
- `HDBExistingBuilding.geojson` - 
- `SLACadastralLandParcel.geojson` - download from [data.gov.sg](https://data.gov.sg/datasets/d_e7395d743076a2bcc487b0d12b9bf33b/view)
- `Generalinformationofschools.csv` - download from [data.gov.sg](https://data.gov.sg/datasets/d_688b934f82c1059ed0a6993d2a829089/view?dataExplorerPage=7)

## How to Run
Run each script one at a time:

```bash
python hdb_amenity_data_cleaning.py
python good_pri_sch_data.py
python dist_bands.py
python <model py>
