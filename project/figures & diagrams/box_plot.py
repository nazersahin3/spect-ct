import pandas as pd
import matplotlib.pyplot as plt


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

df = pd.concat([df_002, df_005, df_006], ignore_index=True)


# ==========================================
# KEEP ONLY DOSE DATA
# ==========================================

dose_df = df[df["roi_type"] == "Dose"].copy()


# ==========================================
# SPLIT PRE AND POST
# ==========================================

pre = dose_df[dose_df["timepoint"] == "PRE"].copy()
post = dose_df[dose_df["timepoint"] == "POST"].copy()

pre = pre[["patient_id", "roi_name", "roi_tlf_pct_min"]]
post = post[["patient_id", "roi_name", "roi_tlf_pct_min"]]

pre = pre.rename(columns={"roi_tlf_pct_min": "pre_function"})
post = post.rename(columns={"roi_tlf_pct_min": "post_function"})


# ==========================================
# MERGE PRE AND POST
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
    (results["pre_function"] - results["post_function"])
    / results["pre_function"]
) * 100


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

results = results.sort_values("roi_name")


# ==========================================
# BOX PLOT
# ==========================================

box_data = [
    results.loc[results["roi_name"] == band, "percent_function_loss"].dropna()
    for band in dose_order
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    box_data,
    tick_labels=dose_order
)

# Overlay individual patient points
for i, band in enumerate(dose_order, start=1):
    y = results.loc[results["roi_name"] == band, "percent_function_loss"]
    x = [i] * len(y)
    plt.plot(x, y, "o")

plt.axhline(y=0, linestyle="--", linewidth=1)
plt.axvline(x=3.5, linestyle="--", linewidth=1)

plt.xlabel("Radiation dose band")
plt.ylabel("% function loss")
plt.title("Percentage regional liver function loss across dose bands")

figure_path = (
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/figures/"
    "function_loss_per_dose_bands_boxplot.png"
)

plt.savefig(figure_path, dpi=300, bbox_inches="tight")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()