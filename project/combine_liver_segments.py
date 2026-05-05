#now combining the segments into one volume 
import os
import SimpleITK as sitk


def main():
    input_folder = "/Users/ne/Desktop/honours/spect-data/NE ALTERED/liver_segmentation_pre004"
    output_path = os.path.join(input_folder, "liver_segments_labelmap.nii.gz")

    segment_files = {
        1: "liver_segment_1.nii.gz",
        2: "liver_segment_2.nii.gz",
        3: "liver_segment_3.nii.gz",
        4: "liver_segment_4.nii.gz",
        5: "liver_segment_5.nii.gz",
        6: "liver_segment_6.nii.gz",
        7: "liver_segment_7.nii.gz",
        8: "liver_segment_8.nii.gz",
    }

    first_path = os.path.join(input_folder, segment_files[1])
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Could not find first segment file:\n{first_path}")

    first_img = sitk.ReadImage(first_path)

    combined = sitk.Image(first_img.GetSize(), sitk.sitkUInt8)
    combined.CopyInformation(first_img)

    occupancy = sitk.Image(first_img.GetSize(), sitk.sitkUInt8)
    occupancy.CopyInformation(first_img)

    total_overlap_voxels = 0

    for label_value, filename in segment_files.items():
        path = os.path.join(input_folder, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing segment file:\n{path}")

        seg = sitk.ReadImage(path)

        if seg.GetSize() != combined.GetSize():
            raise ValueError(f"Size mismatch in {filename}")

        seg_mask = seg > 0
        seg_mask_u8 = sitk.Cast(seg_mask, sitk.sitkUInt8)

        # overlap = voxels already occupied AND also in current segment
        overlap_mask = sitk.And(occupancy > 0, seg_mask)
        overlap_count = int(sitk.GetArrayViewFromImage(sitk.Cast(overlap_mask, sitk.sitkUInt8)).sum())

        if overlap_count > 0:
            print(f"Warning: Segment {label_value} overlaps with previous segments at {overlap_count} voxels.")

        total_overlap_voxels += overlap_count

        # write current segment label into combined map
        combined = sitk.Mask(combined, ~seg_mask) + seg_mask_u8 * label_value

        # update occupancy map
        occupancy = sitk.Cast((occupancy > 0) | seg_mask, sitk.sitkUInt8)

    sitk.WriteImage(combined, output_path)

    print("\nDone.")
    print(f"Saved combined label map to:\n{output_path}")
    print(f"Total overlapping voxels detected: {total_overlap_voxels}")


if __name__ == "__main__":
    main()