from pathlib import Path
import SimpleITK as sitk


def main():
    case_id = "post004"

    out_dir = Path(f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/liver_segmentation_{case_id}")

    liver_mask_path = out_dir / f"liver_{case_id}_mask.nii.gz"
    combined_segments_path = out_dir / f"liver_{case_id}_segments_combined.nii.gz"
    clipped_output_path = out_dir / f"liver_{case_id}_segments_clipped_to_mask.nii.gz"

    if not liver_mask_path.exists():
        raise FileNotFoundError(f"Missing liver mask: {liver_mask_path}")

    if not combined_segments_path.exists():
        raise FileNotFoundError(f"Missing combined segments: {combined_segments_path}")

    liver_mask = sitk.ReadImage(str(liver_mask_path), sitk.sitkUInt8)
    combined_segments = sitk.ReadImage(str(combined_segments_path), sitk.sitkUInt8)

    clipped_segments = sitk.Mask(
        combined_segments,
        sitk.Cast(liver_mask, sitk.sitkUInt8)
    )

    sitk.WriteImage(clipped_segments, str(clipped_output_path))

    print("Saved clipped liver segments to:")
    print(clipped_output_path)


if __name__ == "__main__":
    main()