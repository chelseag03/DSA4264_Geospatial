# =========================================
# 1. Import Libraries
# =========================================

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
import geopandas as gpd
import time
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# =========================================
# 2. Scraping Parameters
# =========================================

headers = {"User-Agent": "Mozilla/5.0"}

# Listing all towns from website
towns = [
    "ang-mo-kio", "bedok", "bishan", "bukit-batok", "bukit-merah",
    "bukit-panjang", "bukit-timah", "central", "choa-chu-kang",
    "clementi", "geylang", "hougang", "jurong-east", "jurong-west",
    "kallang", "marine-parade", "novena", "pasir-ris", "punggol",
    "queenstown", "sembawang", "sengkang", "serangoon", "tampines",
    "toa-payoh", "woodlands", "yishun"
]

# Scrape from 2009–2025 inclusive
years = list(range(2009, 2026))  

# Use the maximum possible phase set seen across years
phase_labels = ["1", "2A(1)", "2A(2)", "2A", "2B", "2C", "2C(S)", "3"]

# =========================================
# 3. Helper Functions For Scraping
# =========================================

def is_school_name(line):
    """
    Checks if line of text contains school name
    """
    if not line:
        return False
    if line.startswith("↳"):
        return False
    if line in {
        "P1 Ballot History", "Secondary Cut-Off Point", "JC Cut-Off Point",
        "Blog Posts", "Download App 📲", "School", "Phase 1", "2A", "2B",
        "2C", "2C(S)", "3"
    }:
        return False
    if "Primary Schools" in line or "P1 Ballot History" in line:
        return False
    if re.fullmatch(r"\d+|-", line):
        return False
    return True

def collect_numeric_lines(lines, start_idx):
    """
    Collect plain numeric / '-' lines until the next label / school / section.
    Returns (values, next_idx)
    """
    values = []
    i = start_idx

    while i < len(lines):
        line = lines[i]

        if line.startswith("↳"):
            break
        if is_school_name(line):
            break

        if re.fullmatch(r"\d+|-", line):
            values.append(line)

        i += 1

    return values, i

# =========================================
# 4. Web Scraping
# =========================================

all_records = []

for year in years:
    for town in towns:
        url = f"https://sgschooling.com/year/{year}/{town}"
        print(f"Scraping {url}")

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text("\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            i = 0
            while i < len(lines):
                # Detect school block: starts with school name followed by Vacancy
                if (
                    i + 1 < len(lines)
                    and is_school_name(lines[i])
                    and lines[i + 1].startswith("↳ Vacancy")
                ):
                    school = lines[i]

                    total_match = re.search(r"\((\d+)\)", lines[i + 1])
                    total_intake = int(total_match.group(1)) if total_match else None

                    # Collect vacancy values
                    vacancy_values, next_i = collect_numeric_lines(lines, i + 2)

                    # Next label should be Applied (number of applicants)
                    applied_values = []
                    if next_i < len(lines) and lines[next_i] == "↳ Applied":
                        applied_values, next_i = collect_numeric_lines(lines, next_i + 1)

                    # Choose the phase set based on how many values were found
                    n_vals = max(len(vacancy_values), len(applied_values))

                    if n_vals == 6:
                        phases = ["1", "2A", "2B", "2C", "2C(S)", "3"]
                    elif n_vals == 7:
                        phases = ["1", "2A(1)", "2A(2)", "2B", "2C", "2C(S)", "3"]
                    elif n_vals == 5:
                        phases = ["1", "2A", "2B", "2C", "3"]
                    else:
                        phases = phase_labels[:n_vals]

                    for j in range(n_vals):
                        all_records.append({
                            "year": year,
                            "town": town,
                            "school": school,
                            "total_intake": total_intake,
                            "phase": phases[j] if j < len(phases) else f"phase_{j+1}",
                            "vacancy": vacancy_values[j] if j < len(vacancy_values) else None,
                            "applied": applied_values[j] if j < len(applied_values) else None,
                            "source_url": url
                        })

                    i = next_i
                else:
                    i += 1

            time.sleep(0.3)

        except Exception as e:
            print(f"Failed for {url}: {e}")

# =========================================
# 5. Save Scraped Data From Website
# =========================================

df = pd.DataFrame(all_records)
df.to_csv("sgschooling_2009_2025.csv", index=False)
print("Saved to sgschooling_2009_2025.csv")

# =========================================
# 6. Load Data & Data Cleaning
# =========================================

df = pd.read_csv('sgschooling_2009_2025.csv')

# strip values to make sure there's no extra spaces
df["phase"] = df["phase"].str.strip()

# keep only Phase 2B and 2C
pri_sch_oversub = df[df["phase"].isin(["2B", "2C"])].copy()

# convert types
pri_sch_oversub["vacancy"] = pd.to_numeric(pri_sch_oversub["vacancy"], errors="coerce")
pri_sch_oversub["applied"] = pd.to_numeric(pri_sch_oversub["applied"], errors="coerce")
pri_sch_oversub["year"] = pd.to_numeric(pri_sch_oversub["year"], errors="coerce")

# remove invalid rows
pri_sch_oversub = pri_sch_oversub.dropna(subset=["vacancy", "applied"])
pri_sch_oversub = pri_sch_oversub[pri_sch_oversub["vacancy"] > 0]

# =========================================
# 7. Demand Calculation
# =========================================

# aggregate by year
school_demand = (
    pri_sch_oversub.groupby(["year", "school"])
    .agg({
        "vacancy": "sum",
        "applied": "sum"
    })
    .reset_index()
)

# add combined oversubscription ratio for 2B & 2C
school_demand["oversubscription"] = (
    school_demand["applied"] / school_demand["vacancy"]
)

school_demand.head()

# convert oversubscription to percentile within each year
# makes demand comparable across years (since raw values may differ year to year)
school_demand["demand_percentile"] = (
    school_demand.groupby("year")["oversubscription"]
    .rank(pct=True)
)

# =========================================
# 8. School-Level Summary Features
# =========================================
"""
For each school, compute:
* avg_demand_percentile: average popularity across years
* demand_percentile_sd: standard deviation to see how much demand fluctuates (volatility)
* n_years: number of observations
"""
school_summary = (
    school_demand.groupby("school", as_index=False)
    .agg(
        avg_demand_percentile=("demand_percentile", "mean"), # average demand percentile
        demand_percentile_sd=("demand_percentile", "std"),
        n_years=("demand_percentile", "count")
    )
)

# handle any NA values
school_summary["demand_percentile_sd"] = school_summary["demand_percentile_sd"].fillna(0)

"""
Stability is defined as the inverse of fluctuation:
* High SD (unstable demand) → worse
* Low SD (stable demand) → better

We negate SD so that all metrics follow the same direction (higher = better)
"""
school_summary["stability_raw"] = -school_summary["demand_percentile_sd"]

# =========================================
# 9. School Location Data
# =========================================
"""
As no prominent primary school has permanent relocation from 2009-2025,
the info from the website which is from Sep 2025 to Dec 2025 is accurate
accross the entire timeframe.
"""
# load data
moe_schools = pd.read_csv("Generalinformationofschools.csv")

# ensuring no random spaces
moe_schools["mainlevel_code"] = moe_schools["mainlevel_code"].str.strip()

# filter for pri schools
moe_pri_sch = moe_schools[
    (moe_schools["mainlevel_code"] == "PRIMARY") |
    (moe_schools["mainlevel_code"] == "MIXED LEVEL (P1-S4)")
].copy()

# select relevant columns
school_loc = moe_pri_sch.loc[:, ["school_name", "address"]]

# =================================================
# 10. Name Standardisaiton & Joining Both Datasets
# =================================================
def make_school_key(s):
    """
    helper function to standaridise school names for joining
    """
    if pd.isna(s):
        return s

    s = str(s).upper().strip()

    # standardise apostrophes / punctuation
    s = s.replace("’", "'")
    s = s.replace("`", "'")
    s = s.replace("&", "AND")

    # remove generic suffixes
    s = re.sub(r"\bPRIMARY SCHOOL\b", "", s)
    s = re.sub(r"\bPRIMARY\b", "", s)
    s = re.sub(r"\bSCHOOL\b", "", s)

    # remove ALL punctuation (including brackets)
    s = re.sub(r"[^\w\s]", "", s)

    # standardise abbreviations
    s = s.replace("ST. ", "ST ")

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s

school_summary["school_key"] = school_summary["school"].apply(make_school_key)
school_loc["school_key"] = school_loc["school_name"].apply(make_school_key)

# join ALL schools in demand dataset to MOE location data
all_school_loc = school_summary.merge(
    school_loc[["school_name", "school_key", "address"]],
    on="school_key",
    how="left"
)

# checking for any unmatched addresses
print("Unmatched addresses:", all_school_loc["address"].isna().sum())
print(all_school_loc[all_school_loc["address"].isna()][["school"]])

"""
From Google search, above 4 unmatched schools are permanently closed,
so can drop.
"""
schools_to_drop = ["Eunos", "Guangyang", "Juying", "Stamford"]

pri_sch_int = all_school_loc[~all_school_loc["school"].isin(schools_to_drop)]


# =========================================
# 11. Geocoding
# =========================================

url = "https://www.onemap.gov.sg/api/common/elastic/search" # OneMap API URL

# Get unique school names
schools = pri_sch_int["school_name"].drop_duplicates().tolist()

rows = []

for s in schools:
    time.sleep(1)
    params = {
        "searchVal": s,
        "returnGeom": "Y",
        "getAddrDetails": "Y",
        "pageNum": 1
    }

    r = requests.get(url, params=params).json()

    if r["results"]:
        res = r["results"][0]

        rows.append({
            "school_name": s,
            "lat": float(res["LATITUDE"]),
            "long": float(res["LONGITUDE"]),
            "full_address": res["ADDRESS"]
        })

coords_df = pd.DataFrame(rows)

# merge coordinates back to main dataset
pri_sch_int2 = pri_sch_int.merge(coords_df, on="school_name", how="left")

# convert to GeoDataFrame
pri_sch_data = gpd.GeoDataFrame(
    pri_sch_int2,
    geometry=gpd.points_from_xy(pri_sch_int2.long, pri_sch_int2.lat),
    crs="EPSG:4326" 
)

# keeping only necessary fields
gdf = pri_sch_data.loc[:, ["school_name",
                            "full_address", "lat", "long", "geometry",
                            "avg_demand_percentile", "demand_percentile_sd", "stability_raw", "n_years"]].copy()

# =========================================
# 12. Nearby Local Supply Calculation
# =========================================
gdf = gdf.to_crs("EPSG:3414")

# Count how many OTHER primary schools are within 1km of each school
supply_counts = []

for idx, row in gdf.iterrows():
    distances = gdf.geometry.distance(row.geometry)
    nearby_count = int(((distances <= 1000) & (distances > 0)).sum())

    supply_counts.append({
        "school_name": row["school_name"],
        "nearby_schools_1km": nearby_count
    })

supply_df = pd.DataFrame(supply_counts)

# merge supply back into school summary
pri_sch_summary = gdf.merge(supply_df, on="school_name", how="left")
pri_sch_summary["nearby_schools_1km"] = pri_sch_summary["nearby_schools_1km"].fillna(0)

# =========================================
# 13. Creating Composite Score
# =========================================
"""
Composite score is computed based on:
* Average demand (how popular the school is)
* Supply-adjusted demand (demand beyond nearby school availability)
* Stability (consistency over time)

Each with equal weights.
"""

# Regress average demand percentile on local supply
# Residual = demand beyond what local supply alone would predict
X = sm.add_constant(pri_sch_summary[["nearby_schools_1km"]])
y = pri_sch_summary["avg_demand_percentile"]

model = sm.OLS(y, X).fit()

pri_sch_summary["supply_adjusted_demand_raw"] = model.resid

# Standardise components
components = pri_sch_summary[[
    "avg_demand_percentile",
    "supply_adjusted_demand_raw",
    "stability_raw"
]].copy()

scaler = StandardScaler()
scaled = scaler.fit_transform(components)

scaled_df = pd.DataFrame(
    scaled,
    index=pri_sch_summary.index,   # keep same index
    columns=["z_demand", "z_supply_adjusted", "z_stability"]
)

# instead of concat, assign directly to avoid duplicate columns on rerun
pri_sch_summary[["z_demand", "z_supply_adjusted", "z_stability"]] = scaled_df

# make composite score for all primary schools
pri_sch_summary["composite_score"] = (
    pri_sch_summary["z_demand"] +
    pri_sch_summary["z_supply_adjusted"] +
    pri_sch_summary["z_stability"]
) / 3

# just in case old duplicate columns already exist from earlier runs
pri_sch_summary = pri_sch_summary.loc[:, ~pri_sch_summary.columns.duplicated()]

# =========================================
# 14. SAP & GEP Features
# =========================================
"""
* SAP schools (Special Assistance Plan)
* GEP schools (Gifted Education Programme)

Reflect recognised academic selectivity and prestige.
"""

# Listing all SAP primary schools and schools offering GEP 
gep_schools = [
    "ANGLO-CHINESE SCHOOL (PRIMARY)",
    "CATHOLIC HIGH SCHOOL",
    "HENRY PARK PRIMARY SCHOOL",
    "NAN HUA PRIMARY SCHOOL",
    "NANYANG PRIMARY SCHOOL",
    "RAFFLES GIRLS' PRIMARY SCHOOL",
    "ROSYTH SCHOOL",
    "ST. HILDA'S PRIMARY SCHOOL",
    "TAO NAN SCHOOL"
]

sap_schools = [
    "AI TONG SCHOOL",
    "CATHOLIC HIGH SCHOOL",
    "CHIJ ST. NICHOLAS GIRLS' SCHOOL",
    "HOLY INNOCENTS' PRIMARY SCHOOL",
    "HONG WEN SCHOOL",
    "KONG HWA SCHOOL",
    "MAHA BODHI SCHOOL",
    "MARIS STELLA HIGH SCHOOL",
    "NAN HUA PRIMARY SCHOOL",
    "NANYANG PRIMARY SCHOOL",
    "PEI CHUN PUBLIC SCHOOL",
    "PEI HWA PRESBYTERIAN PRIMARY SCHOOL",
    "POI CHING SCHOOL",
    "RED SWASTIKA SCHOOL",
    "TAO NAN SCHOOL"
]

# labelling SAP and GEP schools in the data
pri_sch_summary["has_GEP"] = pri_sch_summary["school_name"].isin(gep_schools).astype(int)
pri_sch_summary["is_SAP"] = pri_sch_summary["school_name"].isin(sap_schools).astype(int)

# =========================================
# 15. "Good" Primary School Classification
# =========================================
"""
A primary school is classified as “good” if it satisfies the following conditions:

* Has composite score ≥ threshold, where `threshold = mean + 1 standard deviation`(i.e. significantly above average) AND
* Is an SAP school OR Offers GEP

This is to balance quantitative demand-based performance, and established institutional quality signals.
"""

# calculating threshold
threshold = pri_sch_summary["composite_score"].mean() + pri_sch_summary["composite_score"].std()

# labelling good schools as per definition
pri_sch_summary["good_school"] = (
    (pri_sch_summary["composite_score"] >= threshold) &
    ((pri_sch_summary["is_SAP"] == 1) |
    (pri_sch_summary["has_GEP"] == 1))
).astype(int)

# =========================================
# 16. Final Dataset
# =========================================

# filtering only for good schools for the final dataset
good_pri_sch_data = pri_sch_summary[pri_sch_summary["good_school"] == 1]

# keep only necessary columns
good_pri_sch_data_final = good_pri_sch_data.loc[:, ["school_name", "full_address", 
                                                    "lat", "long", "geometry",
                                                    "composite_score", "is_SAP", "has_GEP"]].copy()
# =========================================
# 16. Export Final Dataset
# =========================================

# save dataframe to CSV
good_pri_sch_data_final.to_csv("good_primary_schools.csv", index=False)
