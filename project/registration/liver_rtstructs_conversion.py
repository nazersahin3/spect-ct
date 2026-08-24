from pathlib import Path
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

patient_id = "002"

ct_dicom_dir = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-002/RT/CT/DICOM"
)

rtstruct_path = Path(
    f"/Users/nana/Desktop/spect-data/PRISM-WM-002/RT/RTSTRUCTS/DICOM/RTSTRUCT1.2.246.352.221.5501261436850139112.3073155848534931878.dcm"
)
target_structure_name = "Liver"
target_roi_number = 33

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