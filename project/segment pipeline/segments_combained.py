from pathlib import Path
import SimpleITK as sitk
import numpy as np


def main():
    case_id = "RT_006"

    out_dir = Path(f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/rt_mask")
    segments_dir = out_dir / f"liver_{case_id}_segments_raw_totalsegmentator"
    output_path = out_dir / f"liver_{case_id}_segments_combined.nii.gz"
    overlap_path = out_dir / f"liver_{case_id}_segment_overlap_map.nii.gz"

    if not segments_dir.exists():
        raise FileNotFoundError(f"Segments folder not found: {segments_dir}")

    reference_img = None
    combined_arr = None
    overlap_count_arr = None

    for i in range(1, 9):
        seg_path = segments_dir / f"liver_segment_{i}.nii.gz"

        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segment file: {seg_path}")

        seg_img = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)
        seg_arr = sitk.GetArrayFromImage(seg_img) > 0

        if reference_img is None:
            reference_img = seg_img
            combined_arr = np.zeros(seg_arr.shape, dtype=np.uint8)
            overlap_count_arr = np.zeros(seg_arr.shape, dtype=np.uint8)
        else:
            if seg_img.GetSize() != reference_img.GetSize():
                raise ValueError(f"Size mismatch for {seg_path}")
            if seg_img.GetSpacing() != reference_img.GetSpacing():
                raise ValueError(f"Spacing mismatch for {seg_path}")
            if seg_img.GetOrigin() != reference_img.GetOrigin():
                raise ValueError(f"Origin mismatch for {seg_path}")
            if seg_img.GetDirection() != reference_img.GetDirection():
                raise ValueError(f"Direction mismatch for {seg_path}")

        overlap_count_arr[seg_arr] += 1

        # Only assign label where currently empty
        # This avoids silently overwriting earlier labels
        new_voxels = seg_arr & (combined_arr == 0)
        combined_arr[new_voxels] = i

    overlap_voxels = np.sum(overlap_count_arr > 1)
    print(f"Total overlapping voxels: {overlap_voxels}")

    combined_img = sitk.GetImageFromArray(combined_arr)
    combined_img.CopyInformation(reference_img)
    sitk.WriteImage(combined_img, str(output_path))

    overlap_img = sitk.GetImageFromArray(overlap_count_arr)
    overlap_img.CopyInformation(reference_img)
    sitk.WriteImage(overlap_img, str(overlap_path))

    print("Saved combined label map to:")
    print(output_path)

    print("Saved overlap map to:")
    print(overlap_path)


if __name__ == "__main__":
    main()