from pathlib import Path
import SimpleITK as sitk
import numpy as np
from scipy import ndimage


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
        / f"{patient_id}_post_segments_from_pre_transfer_filled_to_post_liver.nii.gz"
    )

    liver_img = sitk.ReadImage(str(post_liver_mask_path), sitk.sitkUInt8)
    seg_img = sitk.ReadImage(str(transferred_segments_path), sitk.sitkUInt8)

    liver = sitk.GetArrayFromImage(liver_img) > 0
    seg = sitk.GetArrayFromImage(seg_img).astype(np.uint8)

    seg[~liver] = 0

    labelled = seg > 0
    missing = liver & ~labelled

    print("Before filling:")
    print(f"Post liver voxels: {np.sum(liver)}")
    print(f"Labelled voxels: {np.sum(labelled & liver)}")
    print(f"Missing voxels: {np.sum(missing)}")
    print(f"Coverage: {100 * np.sum(labelled & liver) / np.sum(liver):.2f}%")
    print(f"Labels present: {np.unique(seg)}")

    if np.sum(labelled & liver) == 0:
        raise ValueError("No labels found inside post liver. Cannot fill.")

    _, nearest_indices = ndimage.distance_transform_edt(
        ~labelled,
        return_indices=True
    )

    filled = seg.copy()

    filled[missing] = seg[
        nearest_indices[0][missing],
        nearest_indices[1][missing],
        nearest_indices[2][missing]
    ]

    filled[~liver] = 0

    print()
    print("After filling:")
    print(f"Labelled voxels: {np.sum((filled > 0) & liver)}")
    print(f"Missing voxels: {np.sum(liver & (filled == 0))}")
    print(f"Coverage: {100 * np.sum((filled > 0) & liver) / np.sum(liver):.2f}%")
    print(f"Labels present: {np.unique(filled)}")

    out_img = sitk.GetImageFromArray(filled.astype(np.uint8))
    out_img.CopyInformation(liver_img)
    sitk.WriteImage(out_img, str(output_path))

    print()
    print("Saved filled post segment map to:")
    print(output_path)


if __name__ == "__main__":
    main()