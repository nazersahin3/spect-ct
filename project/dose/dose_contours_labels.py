from pathlib import Path
import csv

import numpy as np
import pydicom
import SimpleITK as sitk

from platipy.dicom.io.rtdose_to_nifti import convert_rtdose


# ============================================================
# USER SETTINGS
# ============================================================

Patient_id= "006"


RTDOSE_FOLDER = Path(
    f"/Users/nana/Desktop/spect-data/PRISM-WM-{Patient_id}/RT/RTDOSE/DICOM"
)

OUTPUT_FOLDER = Path(
    f"/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/{Patient_id}/DOSE_CONTOURS"
)

# Dose contours from 5 to 30 Gy, in 5 Gy increments.
DOSE_THRESHOLDS_GY = [5, 10, 15, 20, 25, 30]


# ============================================================
# FIND THE RTDOSE FILE
# ============================================================

def find_rtdose_file(folder: Path) -> Path:
    """
    Find the single RTDOSE DICOM file inside the input folder.
    """

    if not folder.is_dir():
        raise NotADirectoryError(
            f"RTDOSE folder does not exist:\n{folder}"
        )

    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files were found inside:\n{folder}"
        )

    rtdose_files = []

    for path in files:
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )

            if dataset.get("Modality", "") == "RTDOSE":
                rtdose_files.append(path)

        except Exception:
            continue

    if len(rtdose_files) == 0:
        raise FileNotFoundError(
            "No DICOM file with Modality = RTDOSE was found "
            f"inside:\n{folder}"
        )

    if len(rtdose_files) > 1:
        raise RuntimeError(
            "More than one RTDOSE file was found:\n"
            + "\n".join(str(path) for path in rtdose_files)
        )

    return rtdose_files[0]


# ============================================================
# CREATE NON-OVERLAPPING LABEL MAP
# ============================================================

def create_dose_labelmap(
    dose_image: sitk.Image,
    thresholds_gy: list[float],
) -> tuple[sitk.Image, list[dict]]:
    """
    Create a non-overlapping dose-band label map.

    Labels:
        0 = dose below 5 Gy
        1 = 5 to less than 10 Gy
        2 = 10 to less than 15 Gy
        3 = 15 to less than 20 Gy
        4 = 20 to less than 25 Gy
        5 = 25 to less than 30 Gy
        6 = 30 Gy or greater
    """

    dose_array = sitk.GetArrayFromImage(dose_image)

    # np.digitize assigns:
    #   0 for dose < 5 Gy
    #   1 for 5 <= dose < 10 Gy
    #   ...
    #   6 for dose >= 30 Gy
    label_array = np.digitize(
        dose_array,
        bins=np.asarray(thresholds_gy),
        right=False,
    ).astype(np.uint8)

    label_image = sitk.GetImageFromArray(label_array)
    label_image.CopyInformation(dose_image)

    label_definitions = []

    # Label 0: below first threshold.
    label_definitions.append(
        {
            "label": 0,
            "dose_range": f"dose < {thresholds_gy[0]} Gy",
            "lower_dose_gy": "",
            "upper_dose_gy": thresholds_gy[0],
        }
    )

    # Labels between consecutive thresholds.
    for index in range(len(thresholds_gy) - 1):
        lower = thresholds_gy[index]
        upper = thresholds_gy[index + 1]
        label = index + 1

        label_definitions.append(
            {
                "label": label,
                "dose_range": (
                    f"{lower} <= dose < {upper} Gy"
                ),
                "lower_dose_gy": lower,
                "upper_dose_gy": upper,
            }
        )

    # Final label: at or above 30 Gy.
    label_definitions.append(
        {
            "label": len(thresholds_gy),
            "dose_range": (
                f"dose >= {thresholds_gy[-1]} Gy"
            ),
            "lower_dose_gy": thresholds_gy[-1],
            "upper_dose_gy": "",
        }
    )

    return label_image, label_definitions


# ============================================================
# CREATE CUMULATIVE ISODOSE MASKS
# ============================================================

def create_cumulative_masks(
    dose_image: sitk.Image,
    thresholds_gy: list[float],
    output_folder: Path,
) -> None:
    """
    Create one binary mask for each isodose threshold.

    For example:
        dose_at_least_10Gy.nii.gz contains every voxel
        receiving 10 Gy or more.
    """

    mask_folder = output_folder / "cumulative_isodose_masks"
    mask_folder.mkdir(parents=True, exist_ok=True)

    for threshold in thresholds_gy:
        mask = sitk.Cast(
            dose_image >= float(threshold),
            sitk.sitkUInt8,
        )

        output_path = (
            mask_folder
            / f"dose_at_least_{threshold}Gy.nii.gz"
        )

        sitk.WriteImage(mask, str(output_path))

        voxel_count = int(
            sitk.GetArrayViewFromImage(mask).sum()
        )

        print(
            f"Saved ≥{threshold} Gy mask: "
            f"{voxel_count:,} voxels"
        )


# ============================================================
# SAVE LABEL DEFINITIONS
# ============================================================

def save_label_definitions(
    definitions: list[dict],
    output_path: Path,
) -> None:
    """
    Save a CSV explaining the meaning of each label value.
    """

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "label",
                "dose_range",
                "lower_dose_gy",
                "upper_dose_gy",
            ],
        )

        writer.writeheader()
        writer.writerows(definitions)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    rtdose_file = find_rtdose_file(RTDOSE_FOLDER)

    print(f"RTDOSE file found:\n{rtdose_file}")

    # Read metadata for a safety check.
    dataset = pydicom.dcmread(
        str(rtdose_file),
        stop_before_pixels=True,
        force=True,
    )

    dose_units = str(
        dataset.get("DoseUnits", "UNKNOWN")
    ).upper()

    dose_type = dataset.get("DoseType", "UNKNOWN")
    summation_type = dataset.get(
        "DoseSummationType",
        "UNKNOWN",
    )

    print("\nRTDOSE metadata")
    print("----------------")
    print(f"Dose units: {dose_units}")
    print(f"Dose type: {dose_type}")
    print(f"Summation type: {summation_type}")

    if dose_units != "GY":
        raise ValueError(
            "This RTDOSE file is not recorded in Gy. "
            f"DoseUnits = {dose_units}. Do not apply Gy "
            "thresholds until the dose units are confirmed."
        )

    # Convert the DICOM RTDOSE to a SimpleITK dose image.
    dose_image = convert_rtdose(
        rtdose_file,
        dose_output_path=(
            OUTPUT_FOLDER
            / "continuous_dose_grid_Gy.nii.gz"
        ),
    )

    dose_array = sitk.GetArrayViewFromImage(dose_image)

    print("\nDose-grid information")
    print("---------------------")
    print(f"Image size: {dose_image.GetSize()}")
    print(f"Voxel spacing: {dose_image.GetSpacing()}")
    print(f"Minimum dose: {dose_array.min():.3f} Gy")
    print(f"Maximum dose: {dose_array.max():.3f} Gy")

    # Create the single non-overlapping label map.
    label_image, label_definitions = create_dose_labelmap(
        dose_image=dose_image,
        thresholds_gy=DOSE_THRESHOLDS_GY,
    )

    labelmap_path = (
        OUTPUT_FOLDER
        / "dose_bands_5_to_30Gy.nii.gz"
    )

    sitk.WriteImage(
        label_image,
        str(labelmap_path),
    )

    # Save a table explaining the label values.
    label_csv_path = (
        OUTPUT_FOLDER
        / "dose_band_label_definitions.csv"
    )

    save_label_definitions(
        definitions=label_definitions,
        output_path=label_csv_path,
    )

    # Create conventional cumulative isodose masks.
    create_cumulative_masks(
        dose_image=dose_image,
        thresholds_gy=DOSE_THRESHOLDS_GY,
        output_folder=OUTPUT_FOLDER,
    )

    print("\nCompleted successfully")
    print("----------------------")
    print(f"Dose image:\n{OUTPUT_FOLDER / 'continuous_dose_grid_Gy.nii.gz'}")
    print(f"\nDose-band label map:\n{labelmap_path}")
    print(f"\nLabel definitions:\n{label_csv_path}")


if __name__ == "__main__":
    main()