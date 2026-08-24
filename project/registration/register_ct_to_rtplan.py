# register_ct_to_rtplan.py

from pathlib import Path
import SimpleITK as sitk


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

patient_id = "002"

base_dir = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED")
output_dir = base_dir / patient_id / "registered"
output_dir.mkdir(parents=True, exist_ok=True)

# This is the RTPLAN CT NIfTI you already created from the DICOM series
rtplan_ct_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/002/registered/RTPLAN_CT_converted_from_DICOM.nii.gz")

pre_ct_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/CT - pre #2/003_CT_Liver_3mm_I41s_pre#2.nii"
)

post_ct_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/CT - post #2/CT CT_Liver_3mm_I41s/003_CT_Liver_3mm_I41s.nii.gz"
)


# --------------------------------------------------
# HELPER: PRINT IMAGE INFO
# --------------------------------------------------

def print_image_info(name, image):
    print(f"\n{name}")
    print("  Size:     ", image.GetSize())
    print("  Spacing:  ", image.GetSpacing())
    print("  Origin:   ", image.GetOrigin())
    print("  Direction:", image.GetDirection())


# --------------------------------------------------
# HELPER: CT PREPROCESSING
# --------------------------------------------------

def clamp_ct(image, lower=-1000, upper=1000):
    """
    Clamp CT intensities to a useful HU range for registration.
    This reduces the influence of extreme values.
    """
    return sitk.Clamp(image, lowerBound=lower, upperBound=upper)


# --------------------------------------------------
# REGISTRATION FUNCTION
# --------------------------------------------------

def rigid_register_ct_to_rtplan(fixed_image, moving_image, name):
    """
    Rigidly registers moving_image to fixed_image.

    fixed_image  = RTPLAN CT
    moving_image = PRE or POST CT

    Returns:
        final_transform
        registered_moving_image
    """

    print(f"\nStarting rigid registration: {name}")

    # Clamp CT HU values for more stable registration
    fixed = clamp_ct(fixed_image)
    moving = clamp_ct(moving_image)

    # Initial transform based on image geometry
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()

    # Mutual information is usually good for CT-to-CT registration
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Random sampling speeds things up
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=250,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid: coarse to fine
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed, moving)

    print(f"\nFinished rigid registration: {name}")
    print("  Final metric value:", registration.GetMetricValue())
    print("  Optimizer stop condition:")
    print(" ", registration.GetOptimizerStopConditionDescription())

    # Resample original moving CT onto RTPLAN CT grid
    registered_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        -1000,
        sitk.sitkFloat32,
    )

    return final_transform, registered_image


# --------------------------------------------------
# READ IMAGES
# --------------------------------------------------

fixed_rtplan_ct = sitk.ReadImage(str(rtplan_ct_path), sitk.sitkFloat32)
moving_pre_ct = sitk.ReadImage(str(pre_ct_path), sitk.sitkFloat32)
moving_post_ct = sitk.ReadImage(str(post_ct_path), sitk.sitkFloat32)

print_image_info("FIXED: RTPLAN CT", fixed_rtplan_ct)
print_image_info("MOVING: PRE CT", moving_pre_ct)
print_image_info("MOVING: POST CT", moving_post_ct)


# --------------------------------------------------
# REGISTER PRE CT TO RTPLAN CT
# --------------------------------------------------

pre_transform, pre_ct_registered = rigid_register_ct_to_rtplan(
    fixed_image=fixed_rtplan_ct,
    moving_image=moving_pre_ct,
    name="PRE_to_RTPLAN",
)

pre_output_path = output_dir / "PRE_CT_registered_to_RTPLAN_rigid.nii.gz"
sitk.WriteImage(pre_ct_registered, str(pre_output_path))

print(f"\nSaved registered PRE CT to:")
print(pre_output_path)


# --------------------------------------------------
# REGISTER POST CT TO RTPLAN CT
# --------------------------------------------------

post_transform, post_ct_registered = rigid_register_ct_to_rtplan(
    fixed_image=fixed_rtplan_ct,
    moving_image=moving_post_ct,
    name="POST_to_RTPLAN",
)

post_output_path = output_dir / "POST_CT_registered_to_RTPLAN_rigid.nii.gz"
sitk.WriteImage(post_ct_registered, str(post_output_path))

print(f"\nSaved registered POST CT to:")
print(post_output_path)


# --------------------------------------------------
# SAVE TRANSFORMS
# --------------------------------------------------

pre_transform_path = output_dir / "PRE_to_RTPLAN_rigid_transform.tfm"
post_transform_path = output_dir / "POST_to_RTPLAN_rigid_transform.tfm"

sitk.WriteTransform(pre_transform, str(pre_transform_path))
sitk.WriteTransform(post_transform, str(post_transform_path))

print(f"\nSaved PRE transform to:")
print(pre_transform_path)

print(f"\nSaved POST transform to:")
print(post_transform_path)

print("\nRegistration complete.")