from pathlib import Path
import SimpleITK as sitk
import numpy as np


def main():
    case_id = "post004"

    out_dir = Path(f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/liver_segmentation_{case_id}")

    liver_mask_path = out_dir / f"liver_{case_id}_mask.nii.gz"
    clipped_segments_path = out_dir / f"liver_{case_id}_segments_clipped_to_mask.nii.gz"

    liver_mask_img = sitk.ReadImage(str(liver_mask_path), sitk.sitkUInt8)
    segments_img = sitk.ReadImage(str(clipped_segments_path), sitk.sitkUInt8)

    liver = sitk.GetArrayFromImage(liver_mask_img) > 0
    segments = sitk.GetArrayFromImage(segments_img)

    labelled = segments > 0

    liver_voxels = np.sum(liver)
    labelled_liver_voxels = np.sum(labelled & liver)
    unlabelled_liver_voxels = np.sum(liver & ~labelled)

    coverage = labelled_liver_voxels / liver_voxels if liver_voxels > 0 else 0

    print(f"Case: {case_id}")
    print(f"Whole liver voxels: {liver_voxels}")
    print(f"Couinaud-labelled liver voxels: {labelled_liver_voxels}")
    print(f"Unlabelled liver voxels: {unlabelled_liver_voxels}")
    print(f"Coverage: {coverage * 100:.2f}%")

    if coverage < 0.90:
        print("WARNING: Couinaud segmentation covers less than 90% of the liver.")
        print("Do NOT use this case for segment statistics without review/correction.")
    elif coverage < 0.95:
        print("CAUTION: Couinaud segmentation coverage is borderline. Visually inspect.")
    else:
        print("Coverage looks acceptable.")


if __name__ == "__main__":
    main()