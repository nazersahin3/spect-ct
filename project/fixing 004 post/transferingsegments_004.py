from pathlib import Path
import SimpleITK as sitk
import numpy as np


def check_geometry(img1, img2, name1="image 1", name2="image 2"):
    if img1.GetSize() != img2.GetSize():
        raise ValueError(f"{name1} and {name2} have different sizes.")
    if img1.GetSpacing() != img2.GetSpacing():
        raise ValueError(f"{name1} and {name2} have different spacing.")
    if img1.GetOrigin() != img2.GetOrigin():
        raise ValueError(f"{name1} and {name2} have different origins.")
    if img1.GetDirection() != img2.GetDirection():
        raise ValueError(f"{name1} and {name2} have different directions.")


def main():
    patient_id = "004"

    base_dir = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED")

    pre_case = "pre004"
    post_case = "post004"

    post_ct_path = Path(
        "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/004/post ct 004/3 CT_Liver_3mm_I41s.nii"
    )

    # Use your GOOD pre-treatment Couinaud label map here
    pre_segments_path = ("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/liver_segmentation_pre004/liver_segments_clipped_004_pre.nii.gz")

    post_liver_mask_path = (
        base_dir
        / f"liver_segmentation_{post_case}"
        / f"liver_{post_case}_mask.nii.gz"
    )

    registration_dir = base_dir / f"registration_{patient_id}_pre_to_post_centroid_only"

    transform_path = registration_dir / f"{patient_id}_pre_to_post_centroid_only.tfm"

    transformed_segments_path = (
        registration_dir
        / f"{patient_id}_pre_segments_registered_to_post_centroid_only.nii.gz"
    )

    final_clipped_segments_path = (
        registration_dir
        / f"{patient_id}_pre_segments_registered_to_post_centroid_only_clipped_to_post_liver.nii.gz"
    )

    # ----------------------------
    # Load images
    # ----------------------------
    post_ct = sitk.ReadImage(str(post_ct_path), sitk.sitkFloat32)
    pre_segments = sitk.ReadImage(str(pre_segments_path), sitk.sitkUInt8)
    post_liver_mask = sitk.ReadImage(str(post_liver_mask_path), sitk.sitkUInt8)
    transform = sitk.ReadTransform(str(transform_path))

    post_liver_mask = sitk.BinaryThreshold(post_liver_mask, 1, 255, 1, 0)

    check_geometry(post_liver_mask, post_ct, "post liver mask", "post CT")

    # ----------------------------
    # Resample pre Couinaud labels into post CT space
    # IMPORTANT: nearest-neighbour interpolation for labels
    # ----------------------------
    transformed_segments = sitk.Resample(
        pre_segments,
        post_ct,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    sitk.WriteImage(transformed_segments, str(transformed_segments_path))

    print("Saved transformed pre Couinaud labels to:")
    print(transformed_segments_path)

    # ----------------------------
    # Clip transformed labels to post liver mask
    # ----------------------------
    clipped_segments = sitk.Mask(
        transformed_segments,
        sitk.Cast(post_liver_mask, sitk.sitkUInt8)
    )

    sitk.WriteImage(clipped_segments, str(final_clipped_segments_path))

    print()
    print("Saved transformed labels clipped to post liver mask:")
    print(final_clipped_segments_path)

    # ----------------------------
    # QC stats
    # ----------------------------
    liver_arr = sitk.GetArrayFromImage(post_liver_mask) > 0
    seg_arr = sitk.GetArrayFromImage(clipped_segments)

    labelled = seg_arr > 0

    liver_voxels = np.sum(liver_arr)
    labelled_liver_voxels = np.sum(labelled & liver_arr)
    unlabelled_liver_voxels = np.sum(liver_arr & ~labelled)

    coverage = labelled_liver_voxels / liver_voxels if liver_voxels > 0 else 0

    print()
    print("QC:")
    print(f"Post liver voxels: {liver_voxels}")
    print(f"Labelled post liver voxels: {labelled_liver_voxels}")
    print(f"Unlabelled post liver voxels: {unlabelled_liver_voxels}")
    print(f"Coverage: {coverage * 100:.2f}%")
    print(f"Labels present: {np.unique(seg_arr)}")

    if coverage < 0.75:
        print("WARNING: Coverage is still low. Inspect carefully before using.")
    elif coverage < 0.90:
        print("CAUTION: Coverage is moderate. You may need filling or manual correction.")
    else:
        print("Coverage looks pretty good, but still visually inspect in Slicer.")


if __name__ == "__main__":
    main()