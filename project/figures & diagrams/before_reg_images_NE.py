# ============================================================
# PRE-REGISTRATION LIVER OVERLAY
# Show PRE, POST and RT liver contours together on the RT CT
# ============================================================

import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

from platipy.imaging import ImageVisualiser


patient_id = "006"


# ============================================================
# 1. FILE PATHS
# ============================================================

# ---------------- CT images ----------------

pre_ct_file = (
   "/Users/nana/Desktop/spect-data/PRISM-WM-006/SPECT_nifti_pre/Biliary Scan/3 CT_Liver_3mm_I41s_2.nii"
)

rt_ct_file = (
   "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/registered_RTSTRUCT/RTPLAN_CT_converted_from_DICOM copy.nii.gz"
)

post_ct_file = (
   "/Users/nana/Desktop/spect-data/PRISM-WM-006/SPECT_nifti_post/Biliary Scan/3 CT_Liver_3mm_I41s_3.nii"
)



# ---------------- Liver masks ----------------

pre_liver_mask_file = (
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/liver_segmentation_pre_006/liver_pre_006_mask.nii.gz"
)

post_liver_mask_file = (
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/liver_segmentation_post_006/liver_post_006_mask.nii.gz"
)

rt_liver_mask_file = (
    "/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/006/rtplan_structures/RTPLAN_Liver.nii.gz" # this is the RTSTRUCT liver mask
)


# ---------------- Output ----------------

output_dir = (
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{patient_id}/images"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# LOAD IMAGES
# ============================================================

# CTs - make them all Float32 so SimpleITK is happy
rt_ct = sitk.Cast(
    sitk.ReadImage(rt_ct_file),
    sitk.sitkFloat32
)

pre_ct = sitk.Cast(
    sitk.ReadImage(pre_ct_file),
    sitk.sitkFloat32
)

post_ct = sitk.Cast(
    sitk.ReadImage(post_ct_file),
    sitk.sitkFloat32
)


# Liver masks
rt_liver = sitk.ReadImage(rt_liver_mask_file)
pre_liver = sitk.ReadImage(pre_liver_mask_file)
post_liver = sitk.ReadImage(post_liver_mask_file)


# Check them
for name, img in [
    ("RT CT", rt_ct),
    ("PRE CT", pre_ct),
    ("POST CT", post_ct)
]:
    print(f"\n{name}")
    print("Dimension:", img.GetDimension())
    print("Size:", img.GetSize())
    print("Pixel type:", img.GetPixelIDTypeAsString())


# ============================================================
# INITIAL POSITION ONLY
# ============================================================

pre_initial_transform = sitk.CenteredTransformInitializer(
    rt_ct,
    pre_ct,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

post_initial_transform = sitk.CenteredTransformInitializer(
    rt_ct,
    post_ct,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)


# ============================================================
# 4. PUT PRE AND POST MASKS ONTO THE RT GRID
# ============================================================

pre_liver_rt_space = sitk.Resample(
    pre_liver,                  # image being moved
    rt_ct,                      # RT image defines output grid
    pre_initial_transform,
    sitk.sitkNearestNeighbor,   # IMPORTANT for masks
    0,
    sitk.sitkUInt8
)

post_liver_rt_space = sitk.Resample(
    post_liver,
    rt_ct,
    post_initial_transform,
    sitk.sitkNearestNeighbor,
    0,
    sitk.sitkUInt8
)


# RT liver should already correspond to the RT CT,
# but resampling makes sure the grid matches exactly.

rt_liver_rt_space = sitk.Resample(
    rt_liver,
    rt_ct,
    sitk.Transform(3, sitk.sitkIdentity),
    sitk.sitkNearestNeighbor,
    0,
    sitk.sitkUInt8
)


# ============================================================
# 5. ONE VISUALISER — RT CT BACKGROUND
# ============================================================

vis = ImageVisualiser(
    rt_ct,
    cut=(120, 200, 200)
)


# PRE liver
vis.add_contour(
    pre_liver_rt_space,
    name="Pre Liver",
    color="blue",
    linewidth=2,
    show_legend=True
)


# POST liver
vis.add_contour(
    post_liver_rt_space,
    name="Post Liver",
    color="green",
    linewidth=2,
    show_legend=True
)


# RT liver
vis.add_contour(
    rt_liver_rt_space,
    name="RT Liver",
    color="red",
    linewidth=2,
    show_legend=True
)


# ============================================================
# 6. DISPLAY + SAVE ONE FIGURE
# ============================================================

fig = vis.show()

figure_path = (
    f"{output_dir}/pre_registration_liver_overlay_{patient_id}.png"
)

ax = fig.axes[0]

# Put text in the same upper-right area as the legend
ax.text(
    0.98, 0.98,
    f"Pre-registration liver contours\nPatient {patient_id}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    color="black",
    bbox=dict(
        facecolor="white",
        edgecolor="0.7",
        alpha=0.8,
        boxstyle="round,pad=0.3"
    ),
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("Saved:", figure_path)