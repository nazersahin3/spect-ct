import SimpleITK as sitk
import os

# ----------------------------
# Paths
# ----------------------------
fixed_spect_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/SPECT- pre #2/1000_Mebrofenin_SPECT_F3D_-_AC_pre#2.nii"
moving_spect_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/SPECT - post #2/1000_Mebrofenin_SPECT_F3D_-_AC_post#2.nii"
transform_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/results/ct_post_to_pre_final.h5"

output_dir = "/Users/nana/Desktop/SPECT_registration_results"
os.makedirs(output_dir, exist_ok=True)

print("Fixed SPECT exists:", os.path.exists(fixed_spect_path), fixed_spect_path)
print("Moving SPECT exists:", os.path.exists(moving_spect_path), moving_spect_path)
print("Transform exists:", os.path.exists(transform_path), transform_path)

# ----------------------------
# Load images and transform
# ----------------------------
fixed_spect = sitk.ReadImage(fixed_spect_path, sitk.sitkFloat32)
moving_spect = sitk.ReadImage(moving_spect_path, sitk.sitkFloat32)
final_transform = sitk.ReadTransform(transform_path)

print("Loaded fixed SPECT, moving SPECT, and transform")

# ----------------------------
# Apply transform
# ----------------------------
registered_spect = sitk.Resample(
    moving_spect,
    fixed_spect,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving_spect.GetPixelID()
)

# ----------------------------
# Output paths (two locations)
# ----------------------------
filename = "002_registered_spect_post_to_pre.nii.gz"

project_output_dir = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/results"
desktop_output_dir = "/Users/nana/Desktop/SPECT_registration_results"

os.makedirs(project_output_dir, exist_ok=True)
os.makedirs(desktop_output_dir, exist_ok=True)

project_output_path = os.path.join(project_output_dir, "registered_spect_post_to_pre.nii.gz")
desktop_output_path = os.path.join(desktop_output_dir, "registered_spect_post_to_pre.nii.gz")

# ----------------------------
# Save outputs
# ----------------------------
sitk.WriteImage(registered_spect, project_output_path)
sitk.WriteImage(registered_spect, desktop_output_path)

print("Saved to project:", project_output_path)
print("Saved to desktop:", desktop_output_path)