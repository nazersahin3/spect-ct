import pandas as pd
import matplotlib.pyplot as plt
from platipy.imaging import ImageVisualiser



# ==========================================
# LOAD EACH PATIENT DATAFRAME
# ==========================================

df_002 = pd.read_csv(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/002/evaluation/evaluation_dataframe_evaluated.csv"
)

df_005 = pd.read_csv(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/005/evaluation/evaluation_dataframe_evaluated.csv"
)

df_006 = pd.read_csv(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/evaluation/evaluation_dataframe_evaluated.csv"
)


# ==========================================
# COMBINE THEM
# ==========================================

df = pd.concat(
    [df_002, df_005, df_006],
    ignore_index=True
)

print(df)

# Keep only dose-band rows
dose_df = df[df["roi_type"] == "Dose"].copy()


# ==========================================
# KEEP ONLY DOSE BANDS
# ==========================================

dose_df = df[
    df["roi_type"] == "Dose"
].copy()


# ==========================================
# SEPARATE PRE AND POST
# ==========================================

pre = dose_df[
    dose_df["timepoint"] == "PRE"
].copy()

post = dose_df[
    dose_df["timepoint"] == "POST"
].copy()


# Keep only required columns
pre = pre[
    [
        "patient_id",
        "roi_name",
        "volume_cm3",
        "roi_tlf_pct_min",
    ]
]

post = post[
    [
        "patient_id",
        "roi_name",
        "roi_tlf_pct_min",
    ]
]


# Rename PRE / POST function columns
pre = pre.rename(
    columns={
        "roi_tlf_pct_min": "pre_roi_tlf"
    }
)

post = post.rename(
    columns={
        "roi_tlf_pct_min": "post_roi_tlf"
    }
)


# ==========================================
# MATCH PRE AND POST
# ==========================================

results = pd.merge(
    pre,
    post,
    on=[
        "patient_id",
        "roi_name"
    ]
)


# ==========================================
# CALCULATE FUNCTION LOSS
# ==========================================

results["function_loss"] = (
    results["pre_roi_tlf"]
    - results["post_roi_tlf"]
)

results["function_loss_per_cm3"] = (
    results["function_loss"]
    / results["volume_cm3"]
)


# ==========================================
# DOSE BAND ORDER
# ==========================================

dose_order = [
    "<5 Gy",
    "5-10 Gy",
    "10-15 Gy",
    "15-20 Gy",
    "20-25 Gy",
    "25-30 Gy",
    ">=30 Gy",
]

results["roi_name"] = pd.Categorical(
    results["roi_name"],
    categories=dose_order,
    ordered=True
)

results = results.sort_values(
    [
        "patient_id",
        "roi_name"
    ]
)


# ==========================================
# PLOT ALL PATIENTS
# ==========================================

plt.figure(figsize=(9, 6))


for patient_id, patient_data in results.groupby(
    "patient_id"
):

    plt.plot(
        patient_data["roi_name"],
        patient_data["function_loss_per_cm3"],
        marker="o",
        label=f"Patient {patient_id}"
    )


# ==========================================
# ADD COHORT MEAN
# ==========================================

mean_data = (
    results
    .groupby(
        "roi_name",
        observed=False
    )["function_loss_per_cm3"]
    .mean()
    .reindex(dose_order)
)


plt.plot(
    dose_order,
    mean_data,
    marker="o",
    linewidth=3,
    label="Cohort mean"
)


# ==========================================
# FORMAT
# ==========================================

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=1
)

# Divider at 15 Gy
plt.axvline(
    x=2.5,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Radiation dose band")

plt.ylabel(
    "Functional loss per cm³ (%/min/cm³)"
)

plt.title(
    "Regional liver function loss across radiation dose bands"
)

plt.xticks(rotation=45)

plt.legend()

figure_path = (
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/figures/"
    "function_loss_per_cm3_across_dose_bands_combined_mean.png"
)

plt.savefig(figure_path, dpi=300, bbox_inches="tight")

plt.tight_layout()
plt.show()