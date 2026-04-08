## Feature Engineering
# Run this script to get no. of good schools within each distance band category (<1km, 1-2km, >2km)
# =========================
# Load libraries
# =========================
import pandas as pd
import numpy as np
import re
import requests
import geopandas as gpd
from shapely.ops import nearest_points
import json
import matplotlib.pyplot as plt
#import contextily as cx (only for eda)


# =========================
# Load data
# =========================
hdb_df = pd.read_csv("hdb_with_amenities.csv")
good_sch_df = pd.read_csv("good_primary_schools.csv")

hdb_blocks_gdf = gpd.read_file("HDBExistingBuilding.geojson")
sla_parcel_gdf = gpd.read_file("SLACadastralLandParcel.geojson")

print("HDB blocks CRS:", hdb_blocks_gdf.crs)
print("SLA parcels CRS:", sla_parcel_gdf.crs)


# =========================
# Create matching keys
# =========================
# Resale dataframe keys
hdb_df["block_key"] = hdb_df["block"].astype(str).str.upper().str.strip()
hdb_df["st_cod_key"] = hdb_df["road_code"].astype(str).str.upper().str.strip()

# HDB building polygon keys
hdb_blocks_gdf["block_key"] = hdb_blocks_gdf["BLK_NO"].astype(str).str.upper().str.strip()
hdb_blocks_gdf["st_cod_key"] = hdb_blocks_gdf["ST_COD"].astype(str).str.upper().str.strip()


# =========================
# Create school points
# =========================
# Raw lat/long -> must start in EPSG:4326
good_sch_gdf = gpd.GeoDataFrame(
    good_sch_df.copy(),
    geometry=gpd.points_from_xy(good_sch_df["long"], good_sch_df["lat"]),
    crs="EPSG:4326"
)


# =========================
# Merge HDB resale data with building geometries
# =========================
hdb_with_geom = hdb_df.merge(
    hdb_blocks_gdf[["block_key", "st_cod_key", "geometry"]],
    on=["block_key", "st_cod_key"],
    how="left"
)

# Use the actual CRS of the HDB geometry source
hdb_with_geom = gpd.GeoDataFrame(
    hdb_with_geom,
    geometry="geometry",
    crs=hdb_blocks_gdf.crs
)


# =========================
# Convert all layers to EPSG:3414
# =========================
hdb_poly = hdb_with_geom.to_crs("EPSG:3414").copy()
good_sch_gdf = good_sch_gdf.to_crs("EPSG:3414")
sla_parcel_gdf = sla_parcel_gdf.to_crs("EPSG:3414")


# =========================
# School parcel and SLA joins
# =========================
school_parcel_match = gpd.sjoin(
    good_sch_gdf,
    sla_parcel_gdf,
    how="left",
    predicate="within"
)

print("Matched schools to parcels:", school_parcel_match["LOT_KEY"].notna().sum())
print("Unmatched schools:", school_parcel_match["LOT_KEY"].isna().sum())

school_polygons = school_parcel_match.merge(
    sla_parcel_gdf[["LOT_KEY", "geometry"]],
    on="LOT_KEY",
    how="left",
    suffixes=("_point", "_parcel")
)

# IMPORTANT: use the parcel CRS, not 4326
school_polygons = gpd.GeoDataFrame(
    school_polygons,
    geometry="geometry_parcel",
    crs=sla_parcel_gdf.crs
)

# Keep only rows with parcel geometry
school_polygons = school_polygons[school_polygons.geometry.notna()].copy()

# Optional: fix invalid parcel geometries
school_polygons["geometry"] = school_polygons.geometry.buffer(0)

# Working school polygon layer
sch_poly = school_polygons.reset_index(drop=True).copy()
sch_poly["school_id"] = sch_poly.index


# =========================
# Deduplicate HDB polygons
# =========================
hdb_poly["geom_wkt"] = hdb_poly.geometry.to_wkt()

hdb_unique = hdb_poly[["geom_wkt", "geometry"]].drop_duplicates().reset_index(drop=True)
hdb_unique["hdb_uid"] = hdb_unique.index
hdb_unique["geometry"] = hdb_unique.geometry.buffer(0)


# =========================
# Create buffers
# =========================
hdb_1km = hdb_unique[["hdb_uid", "geometry"]].copy()
hdb_1km["geometry"] = hdb_1km.geometry.buffer(1000)

hdb_2km = hdb_unique[["hdb_uid", "geometry"]].copy()
hdb_2km["geometry"] = hdb_2km.geometry.buffer(2000)

sch_poly = sch_poly.set_geometry("geometry")


# =========================
# Spatial joins
# =========================
join_1km = gpd.sjoin(
    hdb_1km,
    sch_poly[["school_id", "geometry"]],
    how="left",
    predicate="intersects"
)

join_2km = gpd.sjoin(
    hdb_2km,
    sch_poly[["school_id", "geometry"]],
    how="left",
    predicate="intersects"
)


# =========================
# Count schools
# =========================
sch_lt_1km = (
    join_1km.groupby("hdb_uid")["school_id"]
    .nunique()
    .rename("good_sch_lt_1km")
)

sch_lt_2km = (
    join_2km.groupby("hdb_uid")["school_id"]
    .nunique()
    .rename("good_sch_lt_2km")
)

school_feat = (
    pd.concat([sch_lt_1km, sch_lt_2km], axis=1)
    .fillna(0)
    .astype(int)
    .reset_index()
)

school_feat["good_sch_1_2km"] = (
    school_feat["good_sch_lt_2km"] - school_feat["good_sch_lt_1km"]
)

total_schools = sch_poly["school_id"].nunique()
school_feat["good_sch_gt_2km"] = total_schools - school_feat["good_sch_lt_2km"]


# =========================
# Clean old columns
# =========================
school_feature_cols = [
    "good_sch_lt_1km",
    "good_sch_lt_2km",
    "good_sch_1_2km",
    "good_sch_gt_2km",
]

hdb_unique = hdb_unique.drop(columns=school_feature_cols, errors="ignore")
hdb_poly = hdb_poly.drop(columns=school_feature_cols, errors="ignore")


# =========================
# Merge features back to full HDB dataset
# =========================
hdb_unique = hdb_unique.merge(school_feat, on="hdb_uid", how="left")

hdb_poly = hdb_poly.merge(
    hdb_unique[
        [
            "geom_wkt",
            "good_sch_lt_1km",
            "good_sch_lt_2km",
            "good_sch_1_2km",
            "good_sch_gt_2km",
        ]
    ],
    on="geom_wkt",
    how="left"
)


# =========================
# Optional sanity checks
# =========================
print("\n=== CRS CHECK ===")
print("good_sch_gdf CRS:", good_sch_gdf.crs)
print("sla_parcel_gdf CRS:", sla_parcel_gdf.crs)
print("hdb_poly CRS:", hdb_poly.crs)
print("sch_poly CRS:", sch_poly.crs)
print("hdb_1km CRS:", hdb_1km.crs)
print("hdb_2km CRS:", hdb_2km.crs)

print("\n=== GEOMETRY CHECK ===")
print("Missing HDB geometry:", hdb_with_geom.geometry.isna().sum())
print("Missing school parcel geometry:", school_polygons.geometry.isna().sum())
print("Valid HDB geometries:", hdb_poly.geometry.is_valid.sum(), "/", len(hdb_poly))
print("Valid school geometries:", sch_poly.geometry.is_valid.sum(), "/", len(sch_poly))

print("\n=== BOUNDS CHECK ===")
print("HDB bounds:", hdb_poly.total_bounds)
print("School bounds:", sch_poly.total_bounds)

print("\n=== BUFFER AREA CHECK ===")
print("1km sample area:", hdb_1km.geometry.iloc[0].area)
print("2km sample area:", hdb_2km.geometry.iloc[0].area)

print("\n=== FEATURE CHECK ===")
print(school_feat.head(10))
print(
    school_feat[
        ["good_sch_lt_1km", "good_sch_lt_2km", "good_sch_1_2km", "good_sch_gt_2km"]
    ].describe()
)
print(
    "All 2km >= 1km:",
    (school_feat["good_sch_lt_2km"] >= school_feat["good_sch_lt_1km"]).all()
)
print("Any negative 1_2km:", (school_feat["good_sch_1_2km"] < 0).any())

print("\n=== FINAL NULL CHECK ===")
print(
    hdb_poly[
        ["good_sch_lt_1km", "good_sch_lt_2km", "good_sch_1_2km", "good_sch_gt_2km"]
    ].isna().sum()
)

# =========================
# Save to CSV
# =========================
hdb_final_csv = pd.DataFrame(hdb_poly.drop(columns="geometry"))
hdb_final_csv.to_csv("hdb_with_school_features.csv", index=False)


# =========================
# Optional EDA:
# Plot 4-room resale price by number of good schools within 1 km
# =========================
# flat_subset = hdb_poly[hdb_poly["flat_type"] == "4 ROOM"].copy()
#
# fig, ax = plt.subplots(figsize=(8, 5))
# flat_subset.boxplot(
#     column="resale_price",
#     by="sch_1km_bucket",
#     ax=ax,
#     showfliers=False
# )
# ax.set_title("4-room resale price by good schools within 1 km")
# ax.set_xlabel("Good schools within 1 km")
# ax.set_ylabel("Resale price")
# plt.suptitle("")
# plt.show()


# =========================
# Optional EDA:
# Plot HDB polygons by number of good schools within 1 km
# =========================
# plot_gdf = hdb_poly.sample(10000, random_state=42).copy()
#
# plot_gdf["sch_cat"] = pd.cut(
#     plot_gdf["good_sch_lt_1km"],
#     bins=[-1, 0, 1, 10],
#     labels=["0", "1", "2+"]
# )
#
# if plot_gdf.crs is None:
#     plot_gdf = plot_gdf.set_crs(epsg=4326)
# if good_sch_gdf.crs is None:
#     good_sch_gdf = good_sch_gdf.set_crs(epsg=4326)
#
# plot_gdf = plot_gdf.to_crs(epsg=3857)
# good_sch_gdf = good_sch_gdf.to_crs(epsg=3857)
#
# fig, ax = plt.subplots(figsize=(10, 10))
#
# colors = {"0": "#2C7FB8", "1": "#8C564B", "2+": "#1BA3A3"}
#
# for cat in ["0", "1", "2+"]:
#     plot_gdf[plot_gdf["sch_cat"] == cat].plot(
#         ax=ax,
#         color=colors[cat],
#         alpha=0.95,
#         linewidth=0
#     )
#
# good_sch_gdf.plot(ax=ax, color="red", markersize=6, alpha=0.75)
#
# cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, zoom=12)
#
# ax.legend(["Good schools"], loc="lower left")
# ax.set_title("HDB polygons by good school access within 1 km")
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()
