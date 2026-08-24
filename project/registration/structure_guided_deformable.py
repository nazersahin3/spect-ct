from pathlib import Path

import numpy as np
import SimpleITK as sitk

from platipy.imaging.registration.deformable import (
    bspline_registration,
)

from platipy.imaging.registration.utils import (
    apply_transform,
)


# ==================================================
# SETTINGS
# ==================================================

patient_id = "006"

# Run PRE first.
# Change this to "POST" for the POST registration.
timepoint = "POST"

# The metric will be calculated within this distance
# of the exact doctor-delineated liver boundary.
boundary_width_mm = 15.0


# ==================================================
# PATHS
# ==================================================

registration_dir = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/"
    f"NE ALTERED/{patient_id}/registered_RTSTRUCT"
)

rtplan_structures_dir = Path(
    f"/Users/nana/Desktop/HONOURS/spect-data-imac/"
    f"NE ALTERED/{patient_id}/rtplan_structures"
)

output_dir = (
    registration_dir
    / "structure_guided_deformable"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# Fixed planning CT.
fixed_ct_path = (
    registration_dir
    / "RTPLAN_CT_converted_from_DICOM copy.nii.gz"
)

# Fixed doctor-delineated liver.
fixed_liver_path = (
    rtplan_structures_dir
    / "RTPLAN_Liver.nii.gz"
)


# Already rigidly registered moving files.
if timepoint == "PRE":

    moving_ct_path = (
        registration_dir
        / "PRE_CT_to_RTPLAN_structure_rigid.nii.gz"
    )

    moving_liver_path = (
        registration_dir
        / "PRE_liver_to_RTPLAN_structure_rigid.nii.gz"
    )

elif timepoint == "POST":

    moving_ct_path = (
        registration_dir
        / "POST_CT_to_RTPLAN_structure_rigid.nii.gz"
    )

    moving_liver_path = (
        registration_dir
        / "POST_liver_to_RTPLAN_structure_rigid.nii.gz"
    )

else:
    raise ValueError(
        'timepoint must be either "PRE" or "POST".'
    )


# ==================================================
# CHECK INPUTS
# ==================================================

for path in [
    fixed_ct_path,
    fixed_liver_path,
    moving_ct_path,
    moving_liver_path,
]:

    if not path.exists():
        raise FileNotFoundError(
            f"File was not found:\n{path}"
        )


# ==================================================
# READ INPUTS
# ==================================================

fixed_ct = sitk.ReadImage(
    str(fixed_ct_path),
    sitk.sitkFloat32,
)

fixed_liver = sitk.ReadImage(
    str(fixed_liver_path),
)

moving_ct = sitk.ReadImage(
    str(moving_ct_path),
    sitk.sitkFloat32,
)

moving_liver = sitk.ReadImage(
    str(moving_liver_path),
)


# Make sure both structures are binary.
fixed_liver = sitk.Cast(
    fixed_liver > 0,
    sitk.sitkUInt8,
)

moving_liver = sitk.Cast(
    moving_liver > 0,
    sitk.sitkUInt8,
)


# ==================================================
# CHECK IMAGE GRIDS
# ==================================================

def same_grid(image_1, image_2):

    return (
        image_1.GetSize() == image_2.GetSize()
        and np.allclose(
            image_1.GetSpacing(),
            image_2.GetSpacing(),
        )
        and np.allclose(
            image_1.GetOrigin(),
            image_2.GetOrigin(),
        )
        and np.allclose(
            image_1.GetDirection(),
            image_2.GetDirection(),
        )
    )


for image, description in [
    (fixed_liver, "fixed RTPLAN liver"),
    (moving_ct, f"rigid {timepoint} CT"),
    (moving_liver, f"rigid {timepoint} liver"),
]:

    if not same_grid(fixed_ct, image):
        raise RuntimeError(
            f"The {description} is not on the "
            "RTPLAN CT grid."
        )


# ==================================================
# DICE FUNCTION
# ==================================================

def calculate_dice(mask_1, mask_2):

    array_1 = (
        sitk.GetArrayViewFromImage(mask_1) > 0
    )

    array_2 = (
        sitk.GetArrayViewFromImage(mask_2) > 0
    )

    intersection = np.logical_and(
        array_1,
        array_2,
    ).sum()

    denominator = (
        array_1.sum()
        + array_2.sum()
    )

    if denominator == 0:
        return float("nan")

    return float(
        2.0 * intersection / denominator
    )


dice_before = calculate_dice(
    fixed_liver,
    moving_liver,
)


# ==================================================
# CREATE BOUNDARY-DISTANCE GUIDE
# ==================================================

def create_boundary_guide(
    mask,
    distance_limit_mm,
):
    """
    Create a signed distance image.

    Boundary = 0
    Inside liver = positive
    Outside liver = negative
    """

    distance_image = (
        sitk.SignedMaurerDistanceMap(
            mask,
            insideIsPositive=True,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )

    distance_array = sitk.GetArrayFromImage(
        distance_image
    )

    # Limit very distant values so the liver boundary
    # drives the registration.
    guide_array = np.clip(
        distance_array,
        -distance_limit_mm,
        distance_limit_mm,
    )

    # Scale values approximately between -1 and +1.
    guide_array = (
        guide_array
        / distance_limit_mm
    ).astype(np.float32)

    guide_image = sitk.GetImageFromArray(
        guide_array
    )

    guide_image.CopyInformation(mask)

    # Metric-evaluation region centred on the exact
    # fixed liver boundary.
    boundary_band_array = (
        np.abs(distance_array)
        <= distance_limit_mm
    ).astype(np.uint8)

    boundary_band = sitk.GetImageFromArray(
        boundary_band_array
    )

    boundary_band.CopyInformation(mask)

    return guide_image, boundary_band


fixed_liver_guide, fixed_boundary_band = (
    create_boundary_guide(
        fixed_liver,
        boundary_width_mm,
    )
)

moving_liver_guide, _ = (
    create_boundary_guide(
        moving_liver,
        boundary_width_mm,
    )
)


print("\n========================================")
print("BOUNDARY-GUIDED B-SPLINE REGISTRATION")
print("========================================")

print(f"Patient:   {patient_id}")
print(f"Timepoint: {timepoint}")

print(
    f"\nDice after rigid registration: "
    f"{dice_before:.6f}"
)

print(
    "Fixed boundary-band voxels:",
    int(
        np.count_nonzero(
            sitk.GetArrayViewFromImage(
                fixed_boundary_band
            )
        )
    ),
)


# ==================================================
# B-SPLINE REGISTRATION
# ==================================================

print(
    "\nRunning boundary-guided "
    "B-spline registration..."
)

registered_liver_guide, bspline_transform = (
    bspline_registration(

        # Exact RT liver signed-distance guide.
        fixed_image=fixed_liver_guide,

        # Rigid PRE/POST liver signed-distance guide.
        moving_image=moving_liver_guide,

        # Evaluate around the fixed RT liver boundary.
        # Do not supply moving_structure.
        fixed_structure=fixed_boundary_band,

        # Keep your spatial registration parameters.
        resolution_staging=[8, 4, 2],
        smooth_sigmas=[4, 2, 1],

        initial_grid_spacing=64,
        grid_scale_factors=[1, 2, 4],

        # Directly compare corresponding distance values.
        metric="mean_squares",

        # Avoid the LBFGS immediate identity result.
        optimiser="gradient_descent_line_search",

        sampling_rate=0.25,
        number_of_iterations=20,

        interp_order=sitk.sitkLinear,

        # Background value for the distance guide.
        default_value=-1.0,

        isotropic_resample=False,
        verbose=True,
        ncores=4,
    )
)

print("\nB-spline registration completed.")


# ==================================================
# CHECK THE TRANSFORM
# ==================================================

transform_parameters = np.asarray(
    bspline_transform.GetParameters()
)

maximum_coefficient = float(
    np.max(
        np.abs(transform_parameters)
    )
)

mean_coefficient = float(
    np.mean(
        np.abs(transform_parameters)
    )
)

nonzero_coefficients = int(
    np.count_nonzero(
        np.abs(transform_parameters) > 1e-8
    )
)

print("\nTransform check:")

print(
    "Maximum absolute B-spline coefficient:",
    maximum_coefficient,
)

print(
    "Mean absolute B-spline coefficient:",
    mean_coefficient,
)

print(
    "Number of non-zero coefficients:",
    nonzero_coefficients,
)


# Stop rather than silently save another identity output.
if nonzero_coefficients == 0:

    raise RuntimeError(
        "\nThe registration returned an exact identity "
        "transform again. No CT or mask outputs were saved."
    )


# ==================================================
# APPLY TRANSFORM TO LIVER
# ==================================================

deformed_liver = apply_transform(
    input_image=moving_liver,
    reference_image=fixed_ct,
    transform=bspline_transform,
    default_value=0,
    interpolator=sitk.sitkNearestNeighbor,
)

deformed_liver = sitk.Cast(
    deformed_liver > 0,
    sitk.sitkUInt8,
)


# ==================================================
# APPLY TRANSFORM TO CT
# ==================================================

print(
    f"\nApplying transform to rigid "
    f"{timepoint} CT..."
)

deformed_ct = apply_transform(
    input_image=moving_ct,
    reference_image=fixed_ct,
    transform=bspline_transform,
    default_value=-1000,
    interpolator=sitk.sitkLinear,
)


# ==================================================
# RESULTS
# ==================================================

dice_after = calculate_dice(
    fixed_liver,
    deformed_liver,
)

moving_array = (
    sitk.GetArrayViewFromImage(
        moving_liver
    )
)

deformed_array = (
    sitk.GetArrayViewFromImage(
        deformed_liver
    )
)

changed_voxels = int(
    np.count_nonzero(
        moving_array != deformed_array
    )
)


# ==================================================
# SAVE OUTPUTS
# ==================================================

deformed_ct_path = (
    output_dir
    / (
        f"{timepoint}_CT_to_RTPLAN_"
        "boundary_guided_bspline.nii.gz"
    )
)

deformed_liver_path = (
    output_dir
    / (
        f"{timepoint}_liver_to_RTPLAN_"
        "boundary_guided_bspline.nii.gz"
    )
)

transform_path = (
    output_dir
    / (
        f"{timepoint}_to_RTPLAN_"
        "boundary_guided_bspline_transform.tfm"
    )
)

sitk.WriteImage(
    deformed_ct,
    str(deformed_ct_path),
    useCompression=True,
)

sitk.WriteImage(
    deformed_liver,
    str(deformed_liver_path),
    useCompression=True,
)

sitk.WriteTransform(
    bspline_transform,
    str(transform_path),
)


print("\n========================================")
print("REGISTRATION RESULTS")
print("========================================")

print(
    f"Dice after rigid:      "
    f"{dice_before:.6f}"
)

print(
    f"Dice after deformable: "
    f"{dice_after:.6f}"
)

print(
    f"Changed liver voxels:  "
    f"{changed_voxels}"
)

print("\nSaved CT:")
print(deformed_ct_path)

print("\nSaved liver:")
print(deformed_liver_path)

print("\nSaved transform:")
print(transform_path)