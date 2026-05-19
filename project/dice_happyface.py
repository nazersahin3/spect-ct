from pathlib import Path
import csv
import SimpleITK as sitk
import numpy as np


def dice_score(pred_array, ref_array, label):
    pred = pred_array == label
    ref = ref_array == label

    pred_sum = pred.sum()
    ref_sum = ref.sum()

    if pred_sum == 0 and ref_sum == 0:
        return 1.0
    if pred_sum == 0 or ref_sum == 0:
        return 0.0

    intersection = np.logical_and(pred, ref).sum()
    return 2.0 * intersection / (pred_sum + ref_sum)


def check_same_image_space(pred_img, ref_img, case_id):
    if pred_img.GetSize() != ref_img.GetSize():
        raise ValueError(
            f"Size mismatch for {case_id}:\n"
            f"Prediction size: {pred_img.GetSize()}\n"
            f"Reference size: {ref_img.GetSize()}"
        )

    if pred_img.GetSpacing() != ref_img.GetSpacing():
        raise ValueError(
            f"Spacing mismatch for {case_id}:\n"
            f"Prediction spacing: {pred_img.GetSpacing()}\n"
            f"Reference spacing: {ref_img.GetSpacing()}"
        )

    if pred_img.GetOrigin() != ref_img.GetOrigin():
        raise ValueError(
            f"Origin mismatch for {case_id}:\n"
            f"Prediction origin: {pred_img.GetOrigin()}\n"
            f"Reference origin: {ref_img.GetOrigin()}"
        )

    if pred_img.GetDirection() != ref_img.GetDirection():
        raise ValueError(
            f"Direction mismatch for {case_id}:\n"
            f"Prediction direction: {pred_img.GetDirection()}\n"
            f"Reference direction: {ref_img.GetDirection()}"
        )


def main():
    case_numbers = [
        "033",
        "171",
        "213",
        "287",
        "443",
    ]

    results_root = Path(
        "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/MSD DICE"
    )

    reference_mask_root = Path(
        "/Users/nana/Desktop/HONOURS/spect-ct-imac/happyface dataset"
    )

    output_csv = results_root / "dice_results_happyface.csv"

    rows = []

    for case_num in case_numbers:
        case_id = f"MSD_{case_num}"

        pred_path = (
            results_root
            / f"liver_segmentation_{case_id}"
            / f"liver_{case_id}_segments_clipped_to_mask.nii.gz"
        )

        ref_path = reference_mask_root / f"hepaticvessel_{case_num}.nii.gz"

        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction not found: {pred_path}")

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference mask not found: {ref_path}")

        print(f"Processing case {case_id}")
        print("Prediction:", pred_path)
        print("Reference:", ref_path)

        pred_img = sitk.ReadImage(str(pred_path), sitk.sitkUInt8)
        ref_img = sitk.ReadImage(str(ref_path), sitk.sitkUInt8)

        check_same_image_space(pred_img, ref_img, case_id)

        pred = sitk.GetArrayFromImage(pred_img)
        ref = sitk.GetArrayFromImage(ref_img)

        row = {"case_id": case_id}
        dice_values = []

        for label in range(1, 9):
            dsc = dice_score(pred, ref, label)
            row[f"dice_segment_{label}"] = dsc
            dice_values.append(dsc)

        row["mean_dice_segments_1_to_8"] = float(np.mean(dice_values))

        pred_liver = pred > 0
        ref_liver = ref > 0

        intersection = np.logical_and(pred_liver, ref_liver).sum()
        denom = pred_liver.sum() + ref_liver.sum()

        row["dice_whole_liver"] = 2.0 * intersection / denom if denom > 0 else 0.0

        rows.append(row)

    fieldnames = (
        ["case_id"]
        + [f"dice_segment_{i}" for i in range(1, 9)]
        + ["mean_dice_segments_1_to_8", "dice_whole_liver"]
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved Dice results to:")
    print(output_csv)


if __name__ == "__main__":
    main()