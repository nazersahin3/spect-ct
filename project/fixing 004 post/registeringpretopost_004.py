from pathlib import Path
import SimpleITK as sitk


def check_geometry_label(label_img, reference_img, label_name="label"):
    if label_img.GetSize() != reference_img.GetSize():
        raise ValueError(f"{label_name} size does not match reference image.")
    if label_img.GetSpacing() != reference_img.GetSpacing():
        raise ValueError(f"{label_name} spacing does not match reference image.")
    if label_img.GetOrigin() != reference_img.GetOrigin():
        raise ValueError(f"{label_name} origin does not match reference image.")
    if label_img.GetDirection() != reference_img.GetDirection():
        raise ValueError(f"{label_name} direction does not match reference image.")


def print_image_info(name, img):
    print()
    print(name)
    print("  Size:     ", img.GetSize())
    print("  Spacing:  ", img.GetSpacing())
    print("  Origin:   ", img.GetOrigin())
    print("  Direction:", img.GetDirection())


def make_windowed_ct(ct_img):
    return sitk.Clamp(ct_img, lowerBound=-200, upperBound=300)


def main():
    patient_id = "004"

    base_dir = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED")

    pre_case = "pre004"
    post_case = "post004"

    pre_ct_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/004/CT AC AbdoLowDose 3.0 I41s - pre #4/003_AC__AbdoLowDose__3_0__I41s.nii.gz")
    post_ct_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/004/post ct 004/3 CT_Liver_3mm_I41s.nii")

    pre_liver_mask_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/liver_segmentation_pre004/liver_004_pre_mask.nii.gz")
    post_liver_mask_path = Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/liver_segmentation_post004/liver_post004_mask.nii.gz")

    out_dir = base_dir / f"registration_{patient_id}_pre_to_post"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_path = out_dir / f"{patient_id}_pre_to_post_affine.tfm"
    registered_pre_ct_path = out_dir / f"{patient_id}_pre_ct_registered_to_post.nii.gz"

    pre_ct = sitk.ReadImage(str(pre_ct_path), sitk.sitkFloat32)
    post_ct = sitk.ReadImage(str(post_ct_path), sitk.sitkFloat32)

    pre_liver_mask = sitk.ReadImage(str(pre_liver_mask_path), sitk.sitkUInt8)
    post_liver_mask = sitk.ReadImage(str(post_liver_mask_path), sitk.sitkUInt8)

    check_geometry_label(pre_liver_mask, pre_ct, "pre liver mask")
    check_geometry_label(post_liver_mask, post_ct, "post liver mask")

    print_image_info("PRE CT", pre_ct)
    print_image_info("POST CT", post_ct)
    print_image_info("PRE liver mask", pre_liver_mask)
    print_image_info("POST liver mask", post_liver_mask)

    pre_liver_mask = sitk.BinaryThreshold(pre_liver_mask, 1, 255, 1, 0)
    post_liver_mask = sitk.BinaryThreshold(post_liver_mask, 1, 255, 1, 0)

    moving = make_windowed_ct(pre_ct)
    fixed = make_windowed_ct(post_ct)

    # MOMENTS usually gives a better starting point than GEOMETRY
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # Use only the fixed/post liver mask first.
    # This avoids the "all samples outside moving image" issue from an overly strict moving mask.
    registration.SetMetricFixedMask(post_liver_mask)
    # registration.SetMetricMovingMask(pre_liver_mask)

    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.1)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=20
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(initial_transform, inPlace=False)

    print()
    print("Starting affine registration...")

    final_transform = registration.Execute(fixed, moving)

    print("Registration complete.")
    print("Final metric value:", registration.GetMetricValue())
    print("Optimizer stop condition:", registration.GetOptimizerStopConditionDescription())

    sitk.WriteTransform(final_transform, str(transform_path))
    print("Saved transform to:")
    print(transform_path)

    registered_pre_ct = sitk.Resample(
        pre_ct,
        post_ct,
        final_transform,
        sitk.sitkLinear,
        -1024,
        pre_ct.GetPixelID()
    )

    sitk.WriteImage(registered_pre_ct, str(registered_pre_ct_path))

    print("Saved registered pre CT to:")
    print(registered_pre_ct_path)


if __name__ == "__main__":
    main()