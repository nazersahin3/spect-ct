import SimpleITK as sitk
import os

fixed_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/CT - pre #2/003_CT_Liver_3mm_I41s_pre#2.nii"
moving_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/CT - post #2/003_CT_Liver_3mm_I41_post#2.nii"

print("Current working directory:", os.getcwd())
print("Fixed exists:", os.path.exists(fixed_path), fixed_path)
print("Moving exists:", os.path.exists(moving_path), moving_path)

# ----------------------------
# Load images
# ----------------------------
fixed_ct = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
moving_ct = sitk.ReadImage(moving_path, sitk.sitkFloat32)
print("Images loaded")


# ----------------------------
# 1. RIGID REGISTRATION
# ----------------------------
print("Starting rigid registration...")

initial_rigid = sitk.CenteredTransformInitializer(
    fixed_ct,
    moving_ct,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

rigid_reg = sitk.ImageRegistrationMethod()
rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
rigid_reg.SetMetricSamplingStrategy(rigid_reg.RANDOM)
rigid_reg.SetMetricSamplingPercentage(0.01)
rigid_reg.SetInterpolator(sitk.sitkLinear)

rigid_reg.SetOptimizerAsGradientDescent(
    learningRate=0.5,
    numberOfIterations=300,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
rigid_reg.SetOptimizerScalesFromPhysicalShift()

rigid_reg.SetShrinkFactorsPerLevel([4, 2, 1])
rigid_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
rigid_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

rigid_reg.SetInitialTransform(initial_rigid, inPlace=False)
rigid_transform = rigid_reg.Execute(fixed_ct, moving_ct)
print("Rigid done")

# ----------------------------
# 2. AFFINE REGISTRATION
# ----------------------------
print("Starting affine registration...")

affine_initial = sitk.AffineTransform(3)

affine_reg = sitk.ImageRegistrationMethod()
affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
affine_reg.SetMetricSamplingStrategy(affine_reg.RANDOM)
affine_reg.SetMetricSamplingPercentage(0.01)
affine_reg.SetInterpolator(sitk.sitkLinear)

affine_reg.SetOptimizerAsGradientDescent(
    learningRate=0.5,
    numberOfIterations=300,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
affine_reg.SetOptimizerScalesFromPhysicalShift()

affine_reg.SetShrinkFactorsPerLevel([4, 2, 1])
affine_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
affine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

affine_reg.SetMovingInitialTransform(rigid_transform)
affine_reg.SetInitialTransform(affine_initial, inPlace=False)

affine_transform = affine_reg.Execute(fixed_ct, moving_ct)
print("Affine done")


# ----------------------------
# 4. COMPOSE FINAL TRANSFORM - got rid of the bspline
# ----------------------------

final_transform = affine_transform

# ----------------------------
# 5. RESAMPLE MOVING IMAGE TO FIXED
# ----------------------------

print("Resampling moving CT...")

registered_ct = sitk.Resample(
    moving_ct,
    fixed_ct,
    final_transform,
    sitk.sitkLinear,
    0.0,
    fixed_ct.GetPixelID()
)

# ----------------------------
# 6. SAVE OUTPUTS
# ----------------------------
output_dir = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/results"
os.makedirs(output_dir, exist_ok=True)

registered_ct_path = os.path.join(output_dir, "registered_ct_post_to_pre_v2_affine_take3.nii.gz")
final_transform_path = os.path.join(output_dir, "ct_post_to_pre_affine_take3_v2.h5")

sitk.WriteImage(registered_ct, registered_ct_path)
sitk.WriteTransform(final_transform, final_transform_path)

print("Saved registered image to:", registered_ct_path)
print("Saved final transform to:", final_transform_path)
print("Finished")