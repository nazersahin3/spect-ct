import SimpleITK as sitk

# ----------------------------
# Paths
# ----------------------------
spect_post_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/SPECT - post #2/1000_Mebrofenin_SPECT_F3D_-_AC_post#2.nii"
spect_pre_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/SPECT- pre #2/1000_Mebrofenin_SPECT_F3D_-_AC_pre#2.nii"
ct_transform_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/results/ct registration - redo of week 5/take 3/ct_post_to_pre_affine_take3_v2.h5"

# ----------------------------
# Load images
# ----------------------------
spect_post = sitk.ReadImage(spect_post_path, sitk.sitkFloat32)
spect_pre = sitk.ReadImage(spect_pre_path, sitk.sitkFloat32)

# Load CT transform
ct_transform = sitk.ReadTransform(ct_transform_path)

# ----------------------------
# Apply transform
# ----------------------------
registered_spect = sitk.Resample(
    spect_post,          # moving
    spect_pre,           # reference (VERY IMPORTANT)
    ct_transform,
    sitk.sitkLinear,
    0.0,
    spect_post.GetPixelID()
)

# ----------------------------
# Save
# ----------------------------
output_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/results/registered_spect_post_to_pre_from_CT_transform.nii.gz"

sitk.WriteImage(registered_spect, output_path)

print("Saved to:", output_path)
print("Done!")