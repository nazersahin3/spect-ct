import pandas as pd

data = []

patient_id = "006"

dose_bands = [
    "<5 Gy",
    "5-10 Gy",
    "10-15 Gy",
    "15-20 Gy",
    "20-25 Gy",
    "25-30 Gy",
    ">=30 Gy",
]


# ==========================================
# PRE
# ==========================================

for segment in range(1, 9):
    data.append({
        "patient_id": f"{patient_id}",
        "timepoint": "PRE",
        "roi_type": "Couinaud",
        "roi_name": f"Segment {segment}",

        "volume_cm3": None,

        "roi_sum_counts": None,
        "whole_liver_sum_counts": None,
        "roi_fraction_liver_counts": None,

        "patient_tlf_pct_min": None,
        "roi_tlf_pct_min": None,

        "minimum": None,
        "maximum": None,
        "mean": None,
        "standard_deviation": None,
        "percentile_5": None,
        "percentile_95": None,
        "median": None,
    })

for dose_band in dose_bands:
    data.append({
        "patient_id": f"{patient_id}",
        "timepoint": "PRE",
        "roi_type": "Dose",
        "roi_name": dose_band,

        "volume_cm3": None,

        "roi_sum_counts": None,
        "whole_liver_sum_counts": None,
        "roi_fraction_liver_counts": None,

        "patient_tlf_pct_min": None,
        "roi_tlf_pct_min": None,

        "minimum": None,
        "maximum": None,
        "mean": None,
        "standard_deviation": None,
        "percentile_5": None,
        "percentile_95": None,
        "median": None,
    })


# ==========================================
# POST
# ==========================================

for segment in range(1, 9):
    data.append({
        "patient_id": f"{patient_id}",
        "timepoint": "POST",
        "roi_type": "Couinaud",
        "roi_name": f"Segment {segment}",

        "volume_cm3": None,

        "roi_sum_counts": None,
        "whole_liver_sum_counts": None,
        "roi_fraction_liver_counts": None,

        "patient_tlf_pct_min": None,
        "roi_tlf_pct_min": None,

        "minimum": None,
        "maximum": None,
        "mean": None,
        "standard_deviation": None,
        "percentile_5": None,
        "percentile_95": None,
        "median": None,
    })

for dose_band in dose_bands:
    data.append({
        "patient_id": f"{patient_id}",
        "timepoint": "POST",
        "roi_type": "Dose",
        "roi_name": dose_band,

        "volume_cm3": None,

        "roi_sum_counts": None,
        "whole_liver_sum_counts": None,
        "roi_fraction_liver_counts": None,

        "patient_tlf_pct_min": None,
        "roi_tlf_pct_min": None,

        "minimum": None,
        "maximum": None,
        "mean": None,
        "standard_deviation": None,
        "percentile_5": None,
        "percentile_95": None,
        "median": None,
        })


df = pd.DataFrame(data)

print(df)

# Save it
df.to_csv(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/Analysis/evaluation_dataframe_{patient_id}.csv",
    index=False,
)