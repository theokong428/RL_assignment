"""
 Shared data loading & preprocessing

 Prints data summary when run standalone.

"""

import pandas as pd
import numpy as np
import os
import zipfile

# Configuration
NUM_SCENARIOS = 20
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "CaseStudyDataPY")
ZIP_PATH = os.path.join(_SCRIPT_DIR, "CaseStudyDataPY.zip")


def load_all():
    """
    Load and preprocess all data, return as dict.
    """

    #  Extract
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    #  Read raw data
    Suppliers_df = pd.read_csv(f"{DATA_DIR}/Suppliers.csv", index_col=0)
    PostcodeDistricts = pd.read_csv(f"{DATA_DIR}/PostcodeDistricts.csv", index_col=0)
    Candidates_df = pd.read_csv(f"{DATA_DIR}/Candidates.csv", index_col=0)

    DistanceSupplierDistrict_df = pd.read_csv(
        f"{DATA_DIR}/Distance Supplier-District.csv", index_col=0
    )
    DistanceSupplierDistrict_df.columns = DistanceSupplierDistrict_df.columns.astype(int)

    DistanceDistrictDistrict_df = pd.read_csv(
        f"{DATA_DIR}/Distance District-District.csv", index_col=0
    )
    DistanceDistrictDistrict_df.columns = DistanceDistrictDistrict_df.columns.astype(int)

    DemandPeriods_df = pd.read_csv(f"{DATA_DIR}/DemandPeriods.csv")
    DemandPeriods = (
        DemandPeriods_df
        .set_index(["Customer", "Product", "Period"])["Demand"]
        .to_dict()
    )
    nbPeriods = DemandPeriods_df["Period"].max()

    print("Loading scenario demand data...")
    DemandPeriodsScenarios_df = pd.read_csv(f"{DATA_DIR}/DemandPeriodScenarios.csv")
    DemandPeriodsScenarios_df = DemandPeriodsScenarios_df[
        DemandPeriodsScenarios_df["Scenario"] <= NUM_SCENARIOS
    ]
    DemandPeriodsScenarios = (
        DemandPeriodsScenarios_df
        .set_index(["Customer", "Product", "Period", "Scenario"])["Demand"]
        .to_dict()
    )
    print(f"  Loaded {len(DemandPeriodsScenarios)} scenario demand records")

    #  Index sets
    Customers = PostcodeDistricts.index
    Candidates = Candidates_df.index
    Suppliers = Suppliers_df.index
    Products = sorted(DemandPeriods_df["Product"].unique())
    Times = range(1, nbPeriods + 1)
    ScenariosSet = range(1, NUM_SCENARIOS + 1)

    # Vehicle data
    VehicleCostPerMileAndTonneOverall = {1: 0.185, 2: 0.720, 3: 0.857}

    #  Supplier-product mapping
    SuppliersByProduct = {
        p: list(Suppliers_df[Suppliers_df["Product group"] == p].index)
        for p in Products
    }

    # Customer aggregation
    PostalAreas = sorted(PostcodeDistricts["Postal Area"].unique())
    DistrictToArea = PostcodeDistricts["Postal Area"].to_dict()
    AreaToDistricts = {
        area: PostcodeDistricts[PostcodeDistricts["Postal Area"] == area].index.tolist()
        for area in PostalAreas
    }

    # Centroids
    AreaCentroid = {}
    for area in PostalAreas:
        group = PostcodeDistricts.loc[AreaToDistricts[area]]
        AreaCentroid[area] = {
            "x": group["X (Easting)"].mean(),
            "y": group["Y (Northing)"].mean(),
        }

    # Aggregate deterministic demand
    DemandPeriods_df_agg = DemandPeriods_df.copy()
    DemandPeriods_df_agg["Area"] = DemandPeriods_df_agg["Customer"].map(DistrictToArea)
    AggDemandPeriods_df = (
        DemandPeriods_df_agg
        .groupby(["Area", "Product", "Period"])["Demand"]
        .sum()
        .reset_index()
    )
    AggDemandPeriods = (
        AggDemandPeriods_df
        .set_index(["Area", "Product", "Period"])["Demand"]
        .to_dict()
    )

    # Aggregate scenario demand
    DemandScenarios_df_agg = DemandPeriodsScenarios_df.copy()
    DemandScenarios_df_agg["Area"] = DemandScenarios_df_agg["Customer"].map(DistrictToArea)
    AggDemandScenarios_df = (
        DemandScenarios_df_agg
        .groupby(["Area", "Product", "Period", "Scenario"])["Demand"]
        .sum()
        .reset_index()
    )
    AggDemandScenarios = (
        AggDemandScenarios_df
        .set_index(["Area", "Product", "Period", "Scenario"])["Demand"]
        .to_dict()
    )

    # Verify aggregation
    assert DemandPeriods_df["Demand"].sum() == AggDemandPeriods_df["Demand"].sum(), \
        "Demand mismatch after aggregation!"

    # Aggregated distances
    AggDistCandidateCustomer = {}
    for j in Candidates:
        for area in PostalAreas:
            districts = AreaToDistricts[area]
            AggDistCandidateCustomer[(j, area)] = DistanceDistrictDistrict_df.loc[j, districts].mean()

    # Transport costs
    CostSupplierCandidate = {
        (k, j): 2
        * DistanceSupplierDistrict_df.loc[k, j]
        * VehicleCostPerMileAndTonneOverall[Suppliers_df.loc[k, "Vehicle type"]]
        / 1000
        for j in Candidates for k in Suppliers
    }

    CostCandidateCustomer = {
        (j, area): 2
        * AggDistCandidateCustomer[(j, area)]
        * VehicleCostPerMileAndTonneOverall[3]
        / 1000
        for j in Candidates for area in PostalAreas
    }

    # Warehouse and supplier parameters
    f = {j: Candidates_df.loc[j, "Setup cost"] for j in Candidates}
    g = {j: Candidates_df.loc[j, "Operating"] for j in Candidates}
    u = {j: Candidates_df.loc[j, "Capacity"] for j in Candidates}
    s = {k: Suppliers_df.loc[k, "Capacity"] for k in Suppliers}

    # Package and return
    return {
        # DataFrames
        "Suppliers_df": Suppliers_df,
        "PostcodeDistricts": PostcodeDistricts,
        "Candidates_df": Candidates_df,
        # Index sets
        "Customers": Customers,
        "Candidates": Candidates,
        "Suppliers": Suppliers,
        "Products": Products,
        "Times": Times,
        "ScenariosSet": ScenariosSet,
        "PostalAreas": PostalAreas,
        "AreaToDistricts": AreaToDistricts,
        "SuppliersByProduct": SuppliersByProduct,
        # Demand
        "AggDemandPeriods": AggDemandPeriods,
        "AggDemandScenarios": AggDemandScenarios,
        # Costs & parameters
        "CostSupplierCandidate": CostSupplierCandidate,
        "CostCandidateCustomer": CostCandidateCustomer,
        "f": f, "g": g, "u": u, "s": s,
        # Config
        "NUM_SCENARIOS": NUM_SCENARIOS,
        "prob_s": 1.0 / NUM_SCENARIOS,
    }


def print_summary(D):
    """ Print data summary."""
    print("=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"  Customers (original): {len(D['Customers'])}")
    print(f"  Aggregated zones:     {len(D['PostalAreas'])}")
    print(f"  Candidates:           {len(D['Candidates'])}")
    print(f"  Suppliers:            {len(D['Suppliers'])}")
    print(f"  Products:             {len(D['Products'])}")
    print(f"  Periods:              {len(D['Times'])}")
    print(f"  Scenarios:            {D['NUM_SCENARIOS']}")
    print()
    print("Aggregated zones:")
    print(f"  {'Zone':<6} {'Districts':>9} {'Population':>12}")
    for area in D["PostalAreas"]:
        districts = D["AreaToDistricts"][area]
        pop = D["PostcodeDistricts"].loc[districts, "Population"].sum()
        print(f"  {area:<6} {len(districts):>9} {pop:>12,}")
    print()
    for p in D["Products"]:
        print(f"  Product {p}: {len(D['SuppliersByProduct'][p])} suppliers")


if __name__ == "__main__":
    D = load_all()
    print_summary(D)
    print("\nData loaded successfully.")
