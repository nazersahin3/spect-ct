import SimpleITK as sitk

# ----------------------------
# Load images
# ----------------------------
fixed_ct = sitk.ReadImage("project/data/002/CT - pre #2/003_CT_Liver_3mm_I41s_pre#2.nii", sitk.sitkFloat32)
moving_ct = sitk.ReadImage("project/data/002/CT - post #2/003_CT_Liver_3mm_I41_post#2.nii", sitk.sitkFloat32)

# Optional smoothing
fixed_ct = sitk.CurvatureFlow(fixed_ct, timeStep=0.125, numberOfIterations=5)
moving_ct = sitk.CurvatureFlow(moving_ct, timeStep=0.125, numberOfIterations=5)

# ----------------------------
# Downsample for rigid/affine
# ----------------------------
def shrink(image, factor):
    return sitk.Shrink(image, [factor] * image.GetDimension())

fixed_ct_small = shrink(fixed_ct, 4)
moving_ct_small = shrink(moving_ct, 4)

# ----------------------------
# 1. RIGID REGISTRATION
# ----------------------------
initial_rigid = sitk.CenteredTransformInitializer(
    fixed_ct_small,
    moving_ct_small,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

rigid_reg = sitk.ImageRegistrationMethod()
rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
rigid_reg.SetMetricSamplingStrategy(rigid_reg.RANDOM)
rigid_reg.SetMetricSamplingPercentage(0.01)
rigid_reg.SetInterpolator(sitk.sitkLinear)

rigid_reg.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
rigid_reg.SetOptimizerScalesFromPhysicalShift()

rigid_reg.SetShrinkFactorsPerLevel([4, 2, 1])
rigid_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
rigid_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

rigid_reg.SetInitialTransform(initial_rigid, inPlace=False)
rigid_transform = rigid_reg.Execute(fixed_ct_small, moving_ct_small)

# ----------------------------
# 2. AFFINE REGISTRATION
# ----------------------------
affine_initial = sitk.AffineTransform(3)

affine_reg = sitk.ImageRegistrationMethod()
affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
affine_reg.SetMetricSamplingStrategy(affine_reg.RANDOM)
affine_reg.SetMetricSamplingPercentage(0.01)
affine_reg.SetInterpolator(sitk.sitkLinear)

affine_reg.SetOptimizerAsGradientDescent(
    learningRate=0.5,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
affine_reg.SetOptimizerScalesFromPhysicalShift()

affine_reg.SetShrinkFactorsPerLevel([4, 2, 1])
affine_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
affine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# rigid acts as prior alignment
affine_reg.SetMovingInitialTransform(rigid_transform)
affine_reg.SetInitialTransform(affine_initial, inPlace=False)

affine_transform = affine_reg.Execute(fixed_ct_small, moving_ct_small)

# ----------------------------
# 3. BSPLINE REGISTRATION
# ----------------------------
grid_physical_spacing = [40.0, 40.0, 40.0]
image_physical_size = [size * spacing for size, spacing in zip(fixed_ct.GetSize(), fixed_ct.GetSpacing())]

mesh_size = [max(1, int(sz / gsp + 0.5)) for sz, gsp in zip(image_physical_size, grid_physical_spacing)]

initial_bspline = sitk.BSplineTransformInitializer(fixed_ct, mesh_size)

bspline_reg = sitk.ImageRegistrationMethod()
bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
bspline_reg.SetMetricSamplingStrategy(bspline_reg.RANDOM)
bspline_reg.SetMetricSamplingPercentage(0.01)
bspline_reg.SetInterpolator(sitk.sitkLinear)

bspline_reg.SetOptimizerAsLBFGSB(
    gradientConvergenceTolerance=1e-5,
    numberOfIterations=50
)

bspline_reg.SetShrinkFactorsPerLevel([4, 2, 1])
bspline_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# IMPORTANT: affine result is kept as prior alignment
bspline_reg.SetMovingInitialTransform(affine_transform)
bspline_reg.SetInitialTransform(initial_bspline, inPlace=False)

bspline_transform = bspline_reg.Execute(fixed_ct, moving_ct)

# ----------------------------
# 4. COMPOSE FINAL TRANSFORM
# ----------------------------
final_transform = sitk.CompositeTransform(3)
final_transform.AddTransform(rigid_transform)
final_transform.AddTransform(affine_transform)
final_transform.AddTransform(bspline_transform)

# ----------------------------
# 5. RESAMPLE MOVING IMAGE TO FIXED
# ----------------------------
registered_ct = sitk.Resample(
    moving_ct,
    fixed_ct,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving_ct.GetPixelID()
)

sitk.WriteImage(registered_ct, "registered_ct.nii")