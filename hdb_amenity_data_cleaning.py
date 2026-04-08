# -*- coding: utf-8 -*-
"""
Data cleaning and feature engineering pipeline for the HDB resale price project.

Main outputs
------------
1. Cleaned HDB resale dataset
2. MRT/LRT station-exit dataset with opening dates
3. Hawker-centre dataset with location and completion dates
4. Final HDB dataset with nearest-amenity distance features

These outputs directly support the project objective of estimating how
location-related factors affect HDB resale prices.
"""


import pandas as pd
import numpy as np
import re
import json
import geopandas as gpd
import os


# ============================================================
# Section 1: Clean and standardise HDB resale transaction data
# ============================================================

"""
Data sources
------------
- HDB resale transactions: https://data.gov.sg/collections/189/view
- HDB property information:
  https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view
"""

# Load resale transaction files covering the full study period
df1 = pd.read_csv('ResaleFlatPrices (1)/Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv', sep=',')
df2 = pd.read_csv('ResaleFlatPrices (1)/Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv', sep=',')
df3 = pd.read_csv('ResaleFlatPrices (1)/Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv', sep=',')
df4 = pd.read_csv('ResaleFlatPrices (1)/Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv', sep=',')
df5 = pd.read_csv('ResaleFlatPrices (1)/Resale flat prices based on registration date from Jan-2017 onwards.csv', sep=',')

# Standardise structure and formatting across resale source files
def prepare_resale(df):

    df = df.copy()
    columns = [
        "month","town","flat_type","block","street_name","storey_range",
        "floor_area_sqm","flat_model","lease_commence_date","resale_price"
    ]
    df = df[columns]
    df['resale_price'] = df['resale_price'].map(lambda x: float(x))

    # Normalise text fields for accurate merging and comparison
    text_cols = [
        "town","flat_type","block",
        "street_name","storey_range","flat_model"
    ]

    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
    return df

# Clean each source file separately before combining them
resale_dfs = [df1, df2, df3, df4, df5]

cleaned = []

for i, df in enumerate(resale_dfs):
    df = prepare_resale(df)
    df = df.drop_duplicates()
    cleaned.append(df)

# Combining cleaned resale datasets into one main HDB dataset
resale = pd.concat(cleaned, ignore_index=True)

# Load HDB property reference data for validation of existing flats
prop_info = pd.read_csv('HDBPropertyInformation.csv', sep=',')
prop_info = prop_info.iloc[:, :11]
prop_info = prop_info.drop(['year_completed', 'residential', 'commercial', 'max_floor_lvl'], axis=1)

# Prepare join keys in the property reference dataset
prop_info["join_block"] = prop_info["blk_no"].astype(str).str.strip().str.upper()
prop_info["join_street"] = prop_info["street"].astype(str).str.strip().str.upper()

# Prepare equivalent join keys in the resale dataset
resale["join_block"] = resale["block"].astype(str).str.strip().str.upper()
resale["join_street"] = resale["street_name"].astype(str).str.strip().str.upper()

# Keep only resale records that match officially listed existing HDB flats
resale = resale.merge(
    prop_info,
    how="inner",
    on=["join_block","join_street"]
)
resale = resale.drop(columns=["join_block","join_street"])

# Convert transaction month into a time variable suitable for temporal analysis
resale["sold_year_month"] = pd.to_datetime(resale["month"], format="%Y-%m").dt.to_period("M")
resale["sold_year"] = resale["sold_year_month"].dt.year

# Derive lease features used in later housing price modelling
# Assumption: standard HDB lease length is 99 years
resale["sold_remaining_lease"] = 99 - (
    resale["sold_year_month"].dt.year - resale["lease_commence_date"]
)
resale["sold_remaining_lease"] = resale["sold_remaining_lease"].clip(lower=0)

# Create a reference-year lease variable for 2026 to support consistent comparisons
resale["remaining_lease_2026"] = 99 - (2026 - resale["lease_commence_date"])
resale["remaining_lease_2026"] = resale["remaining_lease_2026"].clip(lower=0)

# Drop reference columns that are no longer needed
resale = resale.drop(['blk_no', 'street'], axis=1)


# ============================================================
# Section 2: Map HDB street names to official road codes
# ============================================================
"""
Data sources
------------
- Road name-road code reference obtained from LTA: https://www.lta.gov.sg/content/dam/ltagov/industry_innovations/industry_matters/development_construction_resources/Street_Work_Proposals/Standards_and_Specifications/GIS_Data_Hub/road_name_road_code_jan2024.xlsx
- OneMap road name abbreviations: https://www.onemap.gov.sg/abbreviations/index.html
"""

# Load road code reference table and keep only fields needed for matching

road_code = pd.read_excel( "road_name_road_code_jan2024.xlsx", sheet_name="GISDomain", header=None, skiprows=20)
road_code = road_code[[4, 8]]
road_code.columns = ["road_code", "road_name"]

# Remove incomplete rows before matching
road_code = road_code.dropna().reset_index(drop=True)

# Standardise road-name formatting for duplicate checks
road_code['road_name_clean'] = (
    road_code['road_name']
    .str.upper()
    .str.strip()
)

# Identify road names that map to multiple road codes
dup_roads = road_code['road_name_clean'][road_code['road_name_clean'].duplicated(keep=False)]
print(dup_roads)

# Check whether conflicting road names appear in the HDB resale data
conflict_names = [
    'BRICKLAND CRESCENT',
    'BUANGKOK EAST DRIVE',
    'MILTONIA CLOSE',
    'POH HUAT TERRACE'
]

road_code[road_code['road_name_clean'].isin(conflict_names)] \
    .sort_values(['road_name_clean', 'road_code'])

hdb_conflicts = resale[resale['street_name'].isin(conflict_names)]
hdb_conflicts.info()
## No matching HDB rows were found, so these ambiguous road names can be safely removed

# Remove confirmed non-relevant ambiguous road names from the lookup table
road_code = road_code[
    ~road_code["road_name_full"].isin(conflict_names)
].copy()

# ============================================================
# Section 2.1: Standardise street names using OneMap abbreviations
# ============================================================

# Abbreviation mapping obtained from OneMap
ABBREV_MAP = {
    "ABLY": "ASSEMBLY",
    "ADMIN": "ADMINISTRATION",
    "APT": "APARTMENT",
    "APTS": "APARTMENTS",
    "AVE": "AVENUE",
    "AYE": "AYER RAJAH EXPRESSWAY",
    "BKE": "BUKIT TIMAH EXPRESSWAY",
    "BLDG": "BUILDING",
    "BLK": "BLOCK",
    "BLVD": "BOULEVARD",
    "BO": "BRANCH OFFICE",
    "BR": "BRANCH",
    "BT": "BUKIT",
    "BUDD": "BUDDHIST",
    "CATH": "CATHEDRAL",
    "CBD": "CENTRAL BUSINESS DISTRICT",
    "CC": "COMMUNITY CENTRE/CLUB",
    "CH": "CHURCH",
    "CHBRS": "CHAMBERS",
    "CINE": "CINEMA",
    "CINES": "CINEMAS",
    "CL": "CLOSE",
    "CLUBHSE": "CLUBHOUSE",
    "CONDO": "CONDOMINIUM",
    "CP": "CARPARK",
    "CPLX": "COMPLEX",
    "CRES": "CRESCENT",
    "CT": "COURT",
    "CTE": "CENTRAL EXPRESSWAY",
    "CTR": "CENTRE",
    "CTRL": "CENTRAL",
    "C'WEALTH": "COMMONWEALTH",
    "DEPT": "DEPARTMENT",
    "DEVT": "DEVELOPMENT",
    "DIV": "DIVISION",
    "DR": "DRIVE",
    "ECP": "EAST COAST EXPRESSWAY",
    "EDN": "EDUCATION",
    "ENGRG": "ENGINEERING",
    "ENV": "ENVIRONMENT",
    "ERP": "ELECTRONIC ROAD PRICING",
    "EST": "ESTATE",
    "E'WAY": "EXPRESSWAY",
    "FB": "FOOD BRIDGE",
    "FC": "FOOD CENTRE",
    "FTY": "FACTORY",
    "GDN": "GARDEN",
    "GDNS": "GARDENS",
    "GOVT": "GOVERNMENT",
    "GR": "GROVE",
    "HOSP": "HOSPITAL",
    "HQR": "HEADQUARTER",
    "HQRS": "HEADQUARTERS",
    "HS": "HISTORIC SITE",
    "HSE": "HOUSE",
    "HTS": "HEIGHTS",
    "IND": "INDUSTRIAL",
    "INST": "INSTITUTE",
    "INSTN": "INSTITUTION",
    "INTL": "INTERNATIONAL",
    "JC": "JUNIOR COLLEGES",
    "JLN": "JALAN",
    "JNR": "JUNIOR",
    "KG": "KAMPONG",
    "KJE": "KRANJI EXPRESSWAY",
    "KM": "KILOMETRE",
    "KPE": "KALLANG PAYA LEBAR EXPRESSWAY",
    "LIB": "LIBRARY",
    "LK": "LINK",
    "LOR": "LORONG",
    "MAI": "MAISONETTE",
    "MAIS": "MAISONETTES",
    "MAN": "MANSION",
    "MANS": "MANSIONS",
    "MCE": "MARINA COASTAL EXPRESSWAY",
    "MET": "METROPOLITAN",
    "METH": "METHODIST",
    "MIN": "MINISTY",
    "MJD": "MASJID",
    "MKT": "MARKET",
    "MT": "MOUNT",
    "NATL": "NATIONAL",
    "NPC": "NEIGHBOURHOOD POLICE CENTRES",
    "NPP": "NEIGHBOURHOOD POLICE POSTS",
    "NTH": "NORTH",
    "O/S": "OPEN SPACE",
    "P": "PULAU",
    "P/G": "PLAYGROUND",
    "PIE": "PAN ISLAND EXPRESSWAY",
    "PK": "PARK",
    "PL": "PLACE",
    "POLY": "POLYCLINIC",
    "PRESBY": "PRESBYTERIAN",
    "PRI": "PRIMARY",
    "PT": "POINT",
    "RD": "ROAD",
    "REDEVT": "REDEVELOPMENT",
    "S": "SUNGEI",
    "SCH": "SCHOOL",
    "SEC": "SECONDARY",
    "SLE": "SELETAR EXPRESSWAY",
    "S'PORE": "SINGAPORE",
    "SQ": "SQUARE",
    "ST": "STREET",
    "ST.": "SAINT",
    "STH": "SOUTH",
    "STN": "STATION",
    "TC": "TOWN COUNCIL",
    "TECH": "TECHNICAL",
    "TER": "TERRACE",
    "TG": "TANJONG",
    "TOWNHSE": "TOWNHOUSE",
    "TPE": "TAMPINES EXPRESSWAY",
    "U/C": "UNDER CONSTRUCTION",
    "UPP": "UPPER",
    "VOC": "VOCATIONAL",
    "WAREHSE": "WAREHOUSE",
}

def expand_onemap_abbrev(value):
    """Expand known OneMap abbreviations in a street or road name."""

    if pd.isna(value):
        return pd.NA

    s = str(value).upper().strip()
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()

    # Replace abbreviations token by token to preserve full-word matching
    tokens = s.split(" ")
    expanded = [ABBREV_MAP.get(tok, tok) for tok in tokens]

    s = " ".join(expanded)
    s = re.sub(r"\s+", " ", s).strip()

    return s

# Standardise street names in both datasets before building the road-code lookup
resale["street_name_full"] = resale["street_name"].apply(expand_onemap_abbrev)
road_code["road_name_full"] = road_code["road_name"].apply(expand_onemap_abbrev)

# Build a lookup from standardised road names to official road codes
road_lookup = (
    road_code[["road_name_full", "road_code"]]
    .dropna(subset=["road_name_full", "road_code"])
    .drop_duplicates()
    .set_index("road_name_full")["road_code"]
)

# Map road codes into the resale dataset
resale["road_code"] = resale["street_name_full"].map(road_lookup)

# Create the cleaned HDB dataset used in later stages
hdb_data = resale[["town", "address", "block", "road_code", "street_name",
                     "flat_type", "storey_range", "floor_area_sqm", "flat_model",
                     "lease_commence_date", "resale_price", "sold_year_month",
                     "sold_year", "sold_remaining_lease", "remaining_lease_2026",
                     "market_hawker", "miscellaneous", "multistorey_carpark", "precinct_pavilion"
                     ]]

# Optional export
# hdb_data.to_csv('hdb_data.csv')


# ============================================================
# Section 3: Clean MRT/LRT station-exit data
# ============================================================
"""
Data sources
------------
- MRT/LRT exits: https://data.gov.sg/collections/367/view
- Supplementary station opening data: https://www.kaggle.com/datasets/lzytim/full-list-of-mrt-and-lrt-stations-in-singapore?select=mrt_lrt_stations_2025-01-14.csv
"""

# Load MRT exit GeoJSON and extract station-exit coordinates
with open("LTAMRTStationExitGEOJSON.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

rows = []

for feature in geojson_data.get("features", []):
    geometry = feature.get("geometry", {})
    props = feature.get("properties", {})
    coords = geometry.get("coordinates", [None, None])

    rows.append({
        "station_name": props.get("STATION_NA"),
        "exit_code": props.get("EXIT_CODE"),
        "longitude": coords[0] if len(coords) > 0 else None,
        "latitude": coords[1] if len(coords) > 1 else None
    })

mrt_exit_df = pd.DataFrame(rows)

# Load station opening data and keep relevant fields
mrt_year = pd.read_csv('mrt_lrt_stations_2025-01-14.csv', sep=',')
mrt_year = mrt_year[["station_name", "opening", "type"]]
mrt_year['full_station_name'] = mrt_year["station_name"].str.upper() + " " + mrt_year["type"] + " STATION"

# Convert opening dates into year-month format for consistency
mrt_year['opening'] = pd.to_datetime(mrt_year['opening'], format="ISO8601").dt.to_period("M")

# Add opening information to MRT exit records
mrt_exit_df = mrt_exit_df.merge(
    mrt_year,
    how="left",
    left_on="station_name",
    right_on="full_station_name"
)

# Drop temporary column
mrt_exit_df = mrt_exit_df.drop(columns=["full_station_name"])

# Optional export
# mrt_exit_df.to_csv("mrt_station_exit.csv", index=False)


# ============================================================
# Section 4: Clean hawker-centre data
# ============================================================
"""
Data source
-----------
Hawker centre data: https://data.gov.sg/datasets/d_4a086da0a5553be1d89383cd90d07ecd/view
"""

# Load hawker-centre GeoJSON and extract core fields
with open("HawkerCentresGEOJSON.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for feature in data["features"]:
    coords = feature["geometry"]["coordinates"]
    props = feature["properties"]

    rows.append({
        "hawker_name": props.get("NAME"),
        "hawker_address": props.get("ADDRESS_MYENV"),
        "longitude": coords[0],
        "latitude": coords[1],
        "hup_completion_date": props.get("HUP_COMPLETION_DATE")
    })

hawker_df = pd.DataFrame(rows)

# Convert completion date to datetime
hawker_df["hup_completion_date"] = pd.to_datetime(
    hawker_df["hup_completion_date"],
    format="%d/%m/%Y",
    errors="coerce"
)

# Keep only hawker centres with valid completion dates
hawker_df = hawker_df.dropna(subset=["hup_completion_date"])

# Standardise date format to year-month
hawker_df["hup_completion_date"] = hawker_df["hup_completion_date"].dt.to_period("M")

# Optional export
# hawker_df.to_csv("hawker_centres.csv", index=False)


# ============================================================
# Section 6: Compute nearest-amenity distances for HDB blocks
# ============================================================
"""
Data sources
-----------
- HDB land parcel:

Assume we upload previously created HDB and amenity csv files:
- hdb_data.csv
- mrt_station_exit.csv
- hawker_centres.csv
"""

# Load HDB building polygons
with open("HDBExistingBuilding.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

hdb_blocks_gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

# Load cleaned HDB resale data
hdb_data = pd.read_csv("hdb_data.csv")

# Create standardised keys for linking resale rows to HDB polygons
hdb_data["block_key"] = hdb_data["block"].astype(str).str.upper().str.strip()
hdb_data["st_cod_key"] = hdb_data["road_code"].astype(str).str.upper().str.strip()

hdb_blocks_gdf["block_key"] = hdb_blocks_gdf["BLK_NO"].astype(str).str.upper().str.strip()
hdb_blocks_gdf["st_cod_key"] = hdb_blocks_gdf["ST_COD"].astype(str).str.upper().str.strip()

# Attach building geometry to each resale row
hdb_with_geom = hdb_data.merge(
    hdb_blocks_gdf[["block_key", "st_cod_key", "geometry"]],
    on=["block_key", "st_cod_key"],
    how="left"
)

hdb_with_geom = gpd.GeoDataFrame(hdb_with_geom, geometry="geometry", crs="EPSG:4326")

# Load cleaned amenity datasets
mrt_exit_df = pd.read_csv("mrt_station_exit.csv")
hawker_df = pd.read_csv("hawker_centres.csv")

# Convert amenity tables into GeoDataFrames
mrt_gdf = gpd.GeoDataFrame(
    mrt_exit_df.copy(),
    geometry=gpd.points_from_xy(mrt_exit_df["longitude"], mrt_exit_df["latitude"]),
    crs="EPSG:4326"
)

hawker_gdf = gpd.GeoDataFrame(
    hawker_df.copy(),
    geometry=gpd.points_from_xy(hawker_df["longitude"], hawker_df["latitude"]),
    crs="EPSG:4326"
)

def get_nearest_amenity_for_hdb(
    hdb_gdf,
    amenity_gdf,
    hdb_key_cols,
    amenity_info_cols,
    distance_col_name="amenity_dist",
    distance_crs="EPSG:3414"
):
    """
    Match each unique HDB polygon to its nearest amenity.

    Returns
    -------
    hdb_with_dist_df : pandas.DataFrame
        Original HDB dataframe with the requested distance column added.
    nearest_amenity_df : pandas.DataFrame
        One row per unique HDB polygon with nearest amenity details.
    """

    # Keep only rows with valid geometry
    hdb_valid = hdb_gdf.dropna(subset=["geometry"]).copy()

    # Collapse duplicate resale rows to unique HDB polygons before distance matching
    hdb_unique = (
        hdb_valid[hdb_key_cols + ["geometry"]]
        .drop_duplicates(subset=hdb_key_cols)
        .reset_index(drop=True)
    )

    # Project to a local CRS for accurate distance measurement in metres
    hdb_unique_proj = hdb_unique.to_crs(distance_crs)
    amenity_proj = amenity_gdf.to_crs(distance_crs)

    # Compute nearest amenity using spatial nearest join
    nearest = gpd.sjoin_nearest(
        hdb_unique_proj,
        amenity_proj[amenity_info_cols + ["geometry"]],
        how="left",
        distance_col=distance_col_name
    )

    # Keep only fields needed for downstream use
    nearest_amenity_df = pd.DataFrame(
        nearest[hdb_key_cols + [distance_col_name] + amenity_info_cols]
    )

    # Merge computed distance back into the full HDB dataset
    hdb_with_dist_df = pd.DataFrame(
        hdb_gdf.drop(columns="geometry", errors="ignore").merge(
            nearest_amenity_df[hdb_key_cols + [distance_col_name]],
            on=hdb_key_cols,
            how="left"
        )
    )

    return hdb_with_dist_df, nearest_amenity_df

def get_nearest_mrt_for_hdb(
    hdb_gdf,
    mrt_exit_df,
    hdb_key_cols=["block_key", "st_cod_key"],
    distance_crs="EPSG:3414"
):
    """Return HDB data with nearest MRT distance and a nearest-MRT lookup table."""

    mrt_exit_df = mrt_exit_df.copy()

    # Reconstruct a usable station-name field if prior merges created suffixes
    if "station_name" not in mrt_exit_df.columns:
        if "station_name_y" in mrt_exit_df.columns and "station_name_x" in mrt_exit_df.columns:
            mrt_exit_df["station_name"] = mrt_exit_df["station_name_y"].fillna(mrt_exit_df["station_name_x"])
        elif "station_name_y" in mrt_exit_df.columns:
            mrt_exit_df["station_name"] = mrt_exit_df["station_name_y"]
        elif "station_name_x" in mrt_exit_df.columns:
            mrt_exit_df["station_name"] = mrt_exit_df["station_name_x"]

    # Convert MRT exits into a GeoDataFrame for spatial matching
    mrt_gdf = gpd.GeoDataFrame(
        mrt_exit_df,
        geometry=gpd.points_from_xy(mrt_exit_df["longitude"], mrt_exit_df["latitude"]),
        crs="EPSG:4326"
    )

    # Compute nearest MRT exit for each HDB polygon
    hdb_with_mrt_dist, nearest_mrt_df = get_nearest_amenity_for_hdb(
        hdb_gdf=hdb_gdf,
        amenity_gdf=mrt_gdf,
        hdb_key_cols=hdb_key_cols,
        amenity_info_cols=["station_name", "exit_code", "type"],
        distance_col_name="mrt_dist",
        distance_crs=distance_crs
    )

    return hdb_with_mrt_dist, nearest_mrt_df

def get_nearest_hawker_for_hdb(
    hdb_gdf,
    hawker_df,
    hdb_key_cols=["block_key", "st_cod_key"],
    distance_crs="EPSG:3414"
):
    """Return HDB data with nearest hawker distance and a nearest-hawker lookup table."""

    hawker_df = hawker_df.copy()

    # Convert hawker centres into a GeoDataFrame for spatial matching
    hawker_gdf = gpd.GeoDataFrame(
        hawker_df,
        geometry=gpd.points_from_xy(hawker_df["longitude"], hawker_df["latitude"]),
        crs="EPSG:4326"
    )

    hawker_gdf = hawker_gdf.rename(columns={
        "name": "hawker_name",
        "address": "hawker_address"
    })
    
    # Compute nearest hawker centre for each HDB polygon
    hdb_with_hawker_dist, nearest_hawker_df = get_nearest_amenity_for_hdb(
        hdb_gdf=hdb_gdf,
        amenity_gdf=hawker_gdf,
        hdb_key_cols=hdb_key_cols,
        amenity_info_cols=["hawker_name", "hawker_address"],
        distance_col_name="hawker_dist",
        distance_crs=distance_crs
    )

    return hdb_with_hawker_dist, nearest_hawker_df

# Compute nearest MRT distance features
hdb_with_mrt_dist, nearest_mrt_df = get_nearest_mrt_for_hdb(
    hdb_gdf=hdb_with_geom,
    mrt_exit_df=mrt_exit_df
)

# Compute nearest hawker-centre distance features
hdb_with_hawker_dist, nearest_hawker_df = get_nearest_hawker_for_hdb(
    hdb_gdf=hdb_with_geom,
    hawker_df=hawker_df
)

# Build a one-row-per-block hawker-distance lookup before final merge
hawker_lookup = (
    hdb_with_hawker_dist[["block_key", "st_cod_key", "hawker_dist"]]
    .drop_duplicates(subset=["block_key", "st_cod_key"])
    .copy()
)

# Create tuple keys for compact mapping
hdb_with_mrt_dist = hdb_with_mrt_dist.copy()
hawker_lookup = hawker_lookup.copy()

hdb_with_mrt_dist["merge_key"] = list(zip(hdb_with_mrt_dist["block_key"], hdb_with_mrt_dist["st_cod_key"]))
hawker_lookup["merge_key"] = list(zip(hawker_lookup["block_key"], hawker_lookup["st_cod_key"]))

# Convert hawker lookup to a Series for efficient merge-by-map
hawker_map = hawker_lookup.set_index("merge_key")["hawker_dist"]

# Add hawker distance to the final HDB dataset
hdb_final = hdb_with_mrt_dist.copy()
hdb_final["hawker_dist"] = hdb_final["merge_key"].map(hawker_map)

# Remove temporary merge key after use
hdb_final = hdb_final.drop(columns=["merge_key"])


# ============================================================
# Section 7: Export final datasets
# ============================================================

# Save outputs
base_path = "."
os.makedirs(base_path, exist_ok=True)

hdb_final.to_csv(f"{base_path}/hdb_with_amenities.csv", index=False)
nearest_mrt_df.to_csv(f"{base_path}/nearest_mrt1.csv", index=False)
nearest_hawker_df.to_csv(f"{base_path}/nearest_hawker2.csv", index=False)
