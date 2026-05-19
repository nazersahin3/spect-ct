from pathlib import Path
import SimpleITK as sitk
import numpy as np


def main():
    patient_id = "004"

    base_dir = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED")
    post_case = "post004"

    registration_dir = base_dir / f"registration_{patient_id}_pre_to_post_centroid_only"

    post_liver_mask_path = (
        base_dir
        / f"liver_segmentation_{post_case}"
        / f"liver_{post_case}_mask.nii.gz"
    )

    transferred_segments_path = (
        registration_dir
        / f"{patient_id}_pre_segments_registered_to_post_centroid_only_clipped_to_post_liver.nii.gz"
    )

    output_path = (
        registration_dir
        / f"{patient_id}_unlabelled_post_liver_after_pre_label_transfer.nii.gz"
    )

    liver_img = sitk.ReadImage(str(post_liver_mask_path), sitk.sitkUInt8)
    seg_img = sitk.ReadImage(str(transferred_segments_path), sitk.sitkUInt8)

    liver = sitk.GetArrayFromImage(liver_img) > 0
    seg = sitk.GetArrayFromImage(seg_img) > 0

    unlabelled = liver & ~seg

    out_img = sitk.GetImageFromArray(unlabelled.astype(np.uint8))
    out_img.CopyInformation(liver_img)
    sitk.WriteImage(out_img, str(output_path))

    print("Saved unlabelled post-liver voxels to:")
    print(output_path)
    print(f"Unlabelled voxels: {np.sum(unlabelled)}")
    print(f"Unlabelled fraction: {100 * np.sum(unlabelled) / np.sum(liver):.2f}%")


if __name__ == "__main__":
    main()