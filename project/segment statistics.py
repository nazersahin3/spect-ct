import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# USER INPUTS
# -----------------------------
patient_id = 2
timepoint = "PRE"   # use "PRE" or "POST"

spect_path = Path("/Users/ne/Desktop/honours/spect-data/PRISM-WM-002/SPECT_nifti_pre/Biliary Scan/NM Mebrofenin SPECT F3D AC/1000_Mebrofenin_SPECT_F3D_-_AC.nii.gz")
labelmap_path = Path("/Users/ne/Desktop/honours/spect-data/NE ALTERED/liver_segmentation_pre002/liver_segments_labelmap.nii")

scaling_csv = Path("/Users/ne/Desktop/TLF.csv")
output_csv = Path(f"/Users/ne/Desktop/honours/spect-data/NE ALTERED/liver_segmentation_pre002/{patient_id}_{timepoint}_ROI_TLF_statistics.csv")

# -----------------------------
# READ SCALING SCORES
# -----------------------------
scaling_df = pd.read_csv(scaling_csv)

patient_row = scaling_df.loc[scaling_df["Patient ID"] == patient_id]

if patient_row.empty:
    raise ValueError(f"Patient ID {patient_id} not found in scaling CSV.")

TLF = float(patient_row[timepoint].iloc[0])

print(f"Patient {patient_id} {timepoint} TLF = {TLF} %/min")

# -----------------------------
# READ IMAGES
# -----------------------------
spect_img = sitk.ReadImage(str(spect_path))
label_img = sitk.ReadImage(str(labelmap_path))

# Resample labelmap to SPECT grid if needed
if (
    spect_img.GetSize() != label_img.GetSize()
    or spect_img.GetSpacing() != label_img.GetSpacing()
    or spect_img.GetOrigin() != label_img.GetOrigin()
    or spect_img.GetDirection() != label_img.GetDirection()
):
    print("Resampling labelmap to SPECT grid...")

    label_img = sitk.Resample(
        label_img,
        spect_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label_img.GetPixelID()
    )

spect_arr = sitk.GetArrayFromImage(spect_img).astype(float)
label_arr = sitk.GetArrayFromImage(label_img).astype(int)

# -----------------------------
# WHOLE LIVER COUNTS
# -----------------------------
liver_mask = label_arr > 0
liver_total_counts = np.sum(spect_arr[liver_mask])

print(f"Whole liver total counts = {liver_total_counts}")

# -----------------------------
# VOXEL VOLUME
# -----------------------------
spacing = spect_img.GetSpacing()
voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

# -----------------------------
# ROI STATISTICS + ROI TLF
# -----------------------------
rows = []

labels = sorted([x for x in np.unique(label_arr) if x != 0])

for label in labels:
    mask = label_arr == label
    values = spect_arr[mask]

    roi_total_counts = np.sum(values)
    roi_fraction_counts = roi_total_counts / liver_total_counts
    roi_tlf = roi_fraction_counts * TLF

    voxel_count = values.size
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000

    rows.append({
        "Patient ID": patient_id,
        "Timepoint": timepoint,
        "Segment": f"Segment_{label}",
        "Voxel Count": voxel_count,
        "Volume (mm3)": volume_mm3,
        "Volume (cm3)": volume_cm3,
        "ROI Sum Counts": roi_total_counts,
        "Whole Liver Sum Counts": liver_total_counts,
        "ROI Fraction of Liver Counts": roi_fraction_counts,
        "Patient TLF (%/min)": TLF,
        "ROI_TLF (%/min)": roi_tlf,
        "Minimum": np.min(values),
        "Maximum": np.max(values),
        "Mean": np.mean(values),
        "Standard Deviation": np.std(values),
        "Percentile 5": np.percentile(values, 5),
        "Percentile 95": np.percentile(values, 95),
        "Median": np.median(values),
    })

df = pd.DataFrame(rows)

df.to_csv(output_csv, index=False)

print(df)
print(f"Saved ROI TLF table to: {output_csv}")