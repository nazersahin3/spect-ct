from multiprocessing import freeze_support
from pathlib import Path
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator


def main():
    case_id = "rt_006"

    input_ct = "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/registered/RTPLAN_CT_converted_from_DICOM.nii.gz"
    out_dir = Path(f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/liver_segmentation_{case_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_output_dir = out_dir / f"liver_{case_id}_mask_raw_totalsegmentator"
    ts_output_dir.mkdir(parents=True, exist_ok=True)

    totalsegmentator(
        input=input_ct,
        output=str(ts_output_dir),
        task="total",
        roi_subset=["liver"],
    )

    generated_mask_path = ts_output_dir / "liver.nii.gz"

    if not generated_mask_path.exists():
        raise FileNotFoundError(f"Could not find generated liver mask: {generated_mask_path}")

    liver_mask = sitk.ReadImage(str(generated_mask_path), sitk.sitkUInt8)
    liver_mask = sitk.BinaryThreshold(liver_mask, 1, 255, 1, 0)

    final_mask_path = out_dir / f"liver_{case_id}_mask.nii.gz"
    sitk.WriteImage(liver_mask, str(final_mask_path))

    print("Saved liver mask to:")
    print(final_mask_path)


if __name__ == "__main__":
    freeze_support()
    main()