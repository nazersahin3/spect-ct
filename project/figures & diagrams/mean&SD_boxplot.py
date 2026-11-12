import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# LOAD EACH PATIENT DATAFRAME
# ==========================================

df_002 = pd.read_csv("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/002/evaluation/evaluation_dataframe_evaluated.csv")
df_004 = pd.read_csv("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/evaluation/evaluation_dataframe_evaluated.csv")
df_005 = pd.read_csv("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/005/evaluation/evaluation_dataframe_evaluated.csv")
df_006 = pd.read_csv("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/evaluation/evaluation_dataframe_evaluated.csv")

# Combine all patients
df = pd.concat(
    [df_002, df_004, df_005, df_006],
    ignore_index=True
)


# ==========================================
# KEEP ONLY DOSE-BAND DATA
# ==========================================

dose_df = df[df["roi_type"] == "Dose"].copy()


# ==========================================
# SPLIT PRE AND POST
# ==========================================
# Here I am using the "mean" column for function
# and "standard_deviation" for the SD within that ROI.

pre = dose_df[dose_df["timepoint"] == "PRE"][
    ["patient_id", "roi_name", "mean", "standard_deviation"]
].copy()

post = dose_df[dose_df["timepoint"] == "POST"][
    ["patient_id", "roi_name", "mean", "standard_deviation"]
].copy()

pre = pre.rename(
    columns={
        "mean": "pre_mean",
        "standard_deviation": "pre_sd"
    }
)

post = post.rename(
    columns={
        "mean": "post_mean",
        "standard_deviation": "post_sd"
    }
)


# ==========================================
# MATCH PRE AND POST
# ==========================================

results = pd.merge(
    pre,
    post,
    on=["patient_id", "roi_name"]
)


# ==========================================
# CALCULATE % FUNCTION LOSS
# ==========================================

results["percent_function_loss"] = (
    (results["pre_mean"] - results["post_mean"])
    / results["pre_mean"]
) * 100


# ==========================================
# CALCULATE ERROR BARS (PROPAGATED SD)
# ==========================================

results["percent_function_loss_sd"] = np.sqrt(
    (
        100
        * results["post_mean"]
        / (results["pre_mean"] ** 2)
        * results["pre_sd"]
    ) ** 2
    +
    (
        100
        / results["pre_mean"]
        * results["post_sd"]
    ) ** 2
)


# ==========================================
# ORDER DOSE BANDS
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

results = results.sort_values(["patient_id", "roi_name"])


# ==========================================
# CHECK RESULTS TABLE
# ==========================================

print(results[
    [
        "patient_id",
        "roi_name",
        "percent_function_loss",
        "percent_function_loss_sd"
    ]
])


# ==========================================
# PLOT EACH PATIENT SEPARATELY ON ONE FIGURE
# ==========================================

plt.figure(figsize=(10, 6))

for patient_id, patient_data in results.groupby("patient_id"):
    plt.errorbar(
        patient_data["roi_name"],
        patient_data["percent_function_loss"],
        yerr=patient_data["percent_function_loss_sd"],
        marker="o",
        linestyle="-",
        capsize=4,
        label=f"Patient {patient_id}"
    )


# ==========================================
# FORMATTING
# ==========================================

plt.axhline(y=0, linestyle="--", linewidth=1)
plt.axvline(x=2.5, linestyle="--", linewidth=1)

plt.xlabel("Radiation dose band")
plt.ylabel("% function loss")
plt.title("Percentage regional liver function loss by patient")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()