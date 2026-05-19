from pathlib import Path
import SimpleITK as sitk
import numpy as np


def get_mask_centroid(mask_img):
    stats = sitk.LabelShapeStatisticsImageFilter()
    binary = sitk.BinaryThreshold(mask_img, 1, 255, 1, 0)
    stats.Execute(binary)

    if not stats.HasLabel(1):
        raise ValueError("Mask has no foreground voxels.")

    return stats.GetCentroid(1)


def dice_score(mask_a, mask_b):
    a = sitk.GetArrayFromImage(mask_a) > 0
    b = sitk.GetArrayFromImage(mask_b) > 0

    intersection = np.sum(a & b)
    denom = np.sum(a) + np.sum(b)

    if denom == 0:
        return 0.0

    return 2 * intersection / denom


def main():
    patient_id = "004"

    base_dir = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED")

    pre_case = "pre004"
    post_case = "post004"

    pre_ct_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/004/CT AC AbdoLowDose 3.0 I41s - pre #4/003_AC__AbdoLowDose__3_0__I41s.nii.gz")
    post_ct_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/004/post ct 004/3 CT_Liver_3mm_I41s.nii")

    pre_liver_mask_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/liver_segmentation_pre004/liver_004_pre_mask.nii.gz")
    post_liver_mask_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/liver_segmentation_post004/liver_post004_mask.nii.gz")

    out_dir = base_dir / f"registration_{patient_id}_pre_to_post_centroid_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_path = out_dir / f"{patient_id}_pre_to_post_centroid_only.tfm"
    registered_pre_liver_path = out_dir / f"{patient_id}_pre_liver_mask_registered_to_post_centroid_only.nii.gz"
    registered_pre_ct_path = out_dir / f"{patient_id}_pre_ct_registered_to_post_centroid_only.nii.gz"

    pre_ct = sitk.ReadImage(str(pre_ct_path), sitk.sitkFloat32)
    post_ct = sitk.ReadImage(str(post_ct_path), sitk.sitkFloat32)

    pre_liver_mask = sitk.ReadImage(str(pre_liver_mask_path), sitk.sitkUInt8)
    post_liver_mask = sitk.ReadImage(str(post_liver_mask_path), sitk.sitkUInt8)

    pre_liver_mask = sitk.BinaryThreshold(pre_liver_mask, 1, 255, 1, 0)
    post_liver_mask = sitk.BinaryThreshold(post_liver_mask, 1, 255, 1, 0)

    post_center = get_mask_centroid(post_liver_mask)
    pre_center = get_mask_centroid(pre_liver_mask)

    print("Post liver centroid:", post_center)
    print("Pre liver centroid:", pre_center)

    # SimpleITK registration/resampling transforms map fixed/post space to moving/pre space.
    # So translation = moving centre - fixed centre.
    translation = [
        pre_center[i] - post_center[i]
        for i in range(3)
    ]

    transform = sitk.TranslationTransform(3)
    transform.SetOffset(translation)

    print("Using centroid-only translation:", translation)

    registered_pre_liver = sitk.Resample(
        pre_liver_mask,
        post_liver_mask,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    dice = dice_score(registered_pre_liver, post_liver_mask)

    print(f"Centroid-only Dice: {dice:.3f}")

    sitk.WriteTransform(transform, str(transform_path))
    sitk.WriteImage(registered_pre_liver, str(registered_pre_liver_path))

    registered_pre_ct = sitk.Resample(
        pre_ct,
        post_ct,
        transform,
        sitk.sitkLinear,
        -1024,
        pre_ct.GetPixelID()
    )

    sitk.WriteImage(registered_pre_ct, str(registered_pre_ct_path))

    print("Saved centroid-only transform to:")
    print(transform_path)

    print("Saved registered pre liver mask to:")
    print(registered_pre_liver_path)

    print("Saved registered pre CT to:")
    print(registered_pre_ct_path)


if __name__ == "__main__":
    main()