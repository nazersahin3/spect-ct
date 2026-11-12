import sys
import pydicom

from pathlib import Path
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

patient_id = "004"

ct_dicom_dir = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-004/PRISM_WM_04_08272026/DICOM"
)

rtstruct_path = Path(
    f"/Users/nana/Desktop/spect-data/PRISM-WM-004/PRISM_WM_04_08272026/RTSTRUCT/RTSTRUCT1.2.246.352.221.5489727007827224194.3000301352713694368.dcm"
)
target_structure_name = "Liver"
target_roi_number = 35

output_dir = Path(
    f"/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/{patient_id}/rtplan_structures"
)

output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONVERT RTSTRUCT TO NIFTI MASKS
# --------------------------------------------------

convert_rtstruct(
    dcm_img=ct_dicom_dir,
    dcm_rt_file=rtstruct_path,
    prefix="RTPLAN_",
    output_dir=output_dir,
)

print("\nRTSTRUCT conversion complete.")
print(f"Output folder: {output_dir}")