# register_to_rtplan.py

from pathlib import Path
import SimpleITK as sitk

# Optional but useful for checking DICOM modality
try:
    import pydicom
except ImportError:
    pydicom = None


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

patient_id = "006"

base_dir = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED")

# IMPORTANT:
# This should be the DICOM FOLDER containing the RT planning CT slices,
# not an individual RTPLAN .dcm file.
rtplan_ct_dicom_dir = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-006/RT/CT/3/DICOM"
)

pre_ct_path = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-006/SPECT_nifti_pre/Biliary Scan/3 CT_Liver_3mm_I41s_2.nii"
)

post_ct_path = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-006/SPECT_nifti_post/Biliary Scan/3 CT_Liver_3mm_I41s_3.nii"
)

output_dir = base_dir / patient_id / "registered"
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HELPER: CHECK DICOM MODALITIES
# --------------------------------------------------

def inspect_dicom_modalities(dicom_dir):
    """
    Quickly checks what kind of DICOM files are in the folder.
    We want CT files for the planning CT, not RTPLAN/RTSTRUCT/RTDOSE.
    """
    if pydicom is None:
        print("\n'pydicom' not installed, skipping modality inspection.")
        print("To install: pip install pydicom")
        return

    print("\nInspecting DICOM modalities in:")
    print(dicom_dir)

    modality_counts = {}

    for f in Path(dicom_dir).glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            modality = getattr(ds, "Modality", "UNKNOWN")
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        except Exception as e:
            print(f"Could not read {f.name}: {e}")

    print("\nDICOM modality counts:")
    for modality, count in modality_counts.items():
        print(f"  {modality}: {count}")

    if "CT" not in modality_counts:
        print(
            "\nWARNING: No CT DICOM slices found in this folder.\n"
            "This folder may contain RTPLAN/RTSTRUCT/RTDOSE files instead of the planning CT.\n"
            "You need the folder containing many CT slice DICOMs."
        )


# --------------------------------------------------
# HELPER: READ DICOM CT SERIES
# --------------------------------------------------

def read_dicom_ct_series(dicom_dir):
    """
    Reads a DICOM CT series from a folder as a 3D SimpleITK image.
    """
    dicom_dir = Path(dicom_dir)

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM folder not found: {dicom_dir}")

    inspect_dicom_modalities(dicom_dir)

    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        raise ValueError(
            f"No DICOM image series found in:\n{dicom_dir}\n\n"
            "This usually means the folder does not contain a CT image series. "
            "Check that you are pointing to the folder with CT slice DICOMs, "
            "not a single RTPLAN file."
        )

    print("\nFound DICOM series IDs:")
    for i, series_id in enumerate(series_ids):
        print(f"  [{i}] {series_id}")

    # For now, choose the first series.
    # If there are multiple CT series, we can later modify this to select the correct one.
    selected_series_id = series_ids[0]
    print(f"\nUsing DICOM series ID: {selected_series_id}")

    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), selected_series_id)

    print(f"Number of DICOM slices in selected series: {len(dicom_names)}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    return sitk.Cast(image, sitk.sitkFloat32)


# --------------------------------------------------
# HELPER: PRINT IMAGE INFO
# --------------------------------------------------

def print_image_info(name, image):
    print(f"\n{name}")
    print("  Size:     ", image.GetSize())
    print("  Spacing:  ", image.GetSpacing())
    print("  Origin:   ", image.GetOrigin())
    print("  Direction:", image.GetDirection())


# --------------------------------------------------
# READ IMAGES
# --------------------------------------------------

fixed_rtplan_ct = read_dicom_ct_series(rtplan_ct_dicom_dir)

moving_pre_ct = sitk.ReadImage(str(pre_ct_path), sitk.sitkFloat32)
moving_post_ct = sitk.ReadImage(str(post_ct_path), sitk.sitkFloat32)


# --------------------------------------------------
# PRINT IMAGE INFO
# --------------------------------------------------

print_image_info("FIXED: RTPLAN CT", fixed_rtplan_ct)
print_image_info("MOVING: PRE CT", moving_pre_ct)
print_image_info("MOVING: POST CT", moving_post_ct)


# --------------------------------------------------
#SAVE RTPLAN CT AS NIFTI
# --------------------------------------------------

rtplan_ct_nifti_out = output_dir / "RTPLAN_CT_converted_from_DICOM.nii.gz"

sitk.WriteImage(fixed_rtplan_ct, str(rtplan_ct_nifti_out))

print(f"\nSaved converted RTPLAN CT to:")
print(rtplan_ct_nifti_out)