import SimpleITK as sitk
import numpy as np


def get_centroid(mask):
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)

    return np.array(
        stats.GetCentroid(1)
    )


rt_liver = sitk.ReadImage(
    "/Users/nana/Desktop/HONOURS/spect-data-imac/"
    "NE ALTERED/002/rtplan_structures/"
    "RTPLAN_Liver.nii.gz"
)

pre_liver = sitk.ReadImage(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
    "NE ALTERED/002/liver_segmentation_pre_002/"
    "liver_mask_002_pre.nii"
)


print("\nRTPLAN")
print("Origin:   ", rt_liver.GetOrigin())
print("Spacing:  ", rt_liver.GetSpacing())
print("Direction:", rt_liver.GetDirection())
print("Centroid: ", get_centroid(rt_liver))


print("\nPRE")
print("Origin:   ", pre_liver.GetOrigin())
print("Spacing:  ", pre_liver.GetSpacing())
print("Direction:", pre_liver.GetDirection())
print("Centroid: ", get_centroid(pre_liver))


distance = np.linalg.norm(
    get_centroid(rt_liver)
    -
    get_centroid(pre_liver)
)

print(
    f"\nCentroid separation = "
    f"{distance:.2f} mm"
)