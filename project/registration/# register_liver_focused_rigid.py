# register_liver_focused_rigid.py

from pathlib import Path
import csv

import numpy as np
import SimpleITK as sitk

from platipy.imaging.registration.utils import (
    convert_mask_to_reg_structure,
)


# ==================================================
# Patient Details
# ==================================================

patient_id = "006"

base_dir = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED"
)

output_dir = base_dir / patient_id / "registered_RTSTRUCT"
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FIXED RTPLAN CT AND LIVER MASK
# --------------------------------------------------

rtplan_ct_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/registered/RTPLAN_CT_converted_from_DICOM.nii.gz"
    )

# Replace with the TotalSegmentator liver mask
# generated from the RTPLAN CT.
rtplan_liver_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/006/rtplan_structures/RTPLAN_Liver.nii.gz"
)


# --------------------------------------------------
# ORIGINAL PRE AND POST CTs
# --------------------------------------------------

pre_ct_path = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-006/SPECT_nifti_pre/Biliary Scan/3 CT_Liver_3mm_I41s_2.nii"
)

post_ct_path = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-006/SPECT_nifti_post/Biliary Scan/3 CT_Liver_3mm_I41s_3.nii"
)

# --------------------------------------------------
# ORIGINAL PRE AND POST LIVER MASKS
#
# Each mask must have been generated from the exact
# corresponding CT listed above.
# --------------------------------------------------

pre_liver_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/liver_segmentation_pre_006/liver_pre_006_mask.nii.gz"
)

post_liver_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/006/liver_segmentation_post_006/liver_post_006_mask.nii.gz"
)


# ==================================================
# SETTINGS
# ==================================================

# Amount of surrounding anatomy included in the CT metric.
LIVER_MARGIN_MM = 30.0

# Slight mask expansion for smoother structure images.
STRUCTURE_EXPANSION = (3, 3, 3)

# CT range used during local intensity registration.
CT_LOWER_HU = -200
CT_UPPER_HU = 400


# ==================================================
# BASIC HELPERS
# ==================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def binarise(mask: sitk.Image) -> sitk.Image:
    return sitk.Cast(mask > 0, sitk.sitkUInt8)


def same_geometry(
    image_a: sitk.Image,
    image_b: sitk.Image,
    tolerance: float = 1e-5,
) -> bool:
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(
            image_a.GetSpacing(),
            image_b.GetSpacing(),
            atol=tolerance,
        )
        and np.allclose(
            image_a.GetOrigin(),
            image_b.GetOrigin(),
            atol=tolerance,
        )
        and np.allclose(
            image_a.GetDirection(),
            image_b.GetDirection(),
            atol=tolerance,
        )
    )


def check_mask_matches_ct(
    mask: sitk.Image,
    ct: sitk.Image,
    description: str,
) -> None:
    if not same_geometry(mask, ct):
        raise ValueError(
            f"{description} liver mask does not match its source CT.\n"
            "The mask must be generated from the exact CT used here."
        )


def clamp_ct(image: sitk.Image) -> sitk.Image:
    return sitk.Clamp(
        sitk.Cast(image, sitk.sitkFloat32),
        lowerBound=CT_LOWER_HU,
        upperBound=CT_UPPER_HU,
    )


def mm_to_voxel_radius(
    image: sitk.Image,
    margin_mm: float,
) -> list[int]:
    return [
        max(1, int(np.ceil(margin_mm / spacing)))
        for spacing in image.GetSpacing()
    ]


def make_metric_mask(
    liver_mask: sitk.Image,
    margin_mm: float,
) -> sitk.Image:
    radius = mm_to_voxel_radius(
        liver_mask,
        margin_mm,
    )

    expanded = sitk.BinaryDilate(
        binarise(liver_mask),
        radius,
        sitk.sitkBall,
    )

    return sitk.Cast(expanded, sitk.sitkUInt8)


# ==================================================
# RESAMPLING AND QC
# ==================================================

def resample_ct(
    moving_ct: sitk.Image,
    fixed_ct: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    return sitk.Resample(
        moving_ct,
        fixed_ct,
        transform,
        sitk.sitkLinear,
        -1000,
        sitk.sitkFloat32,
    )


def resample_mask(
    moving_mask: sitk.Image,
    fixed_ct: sitk.Image,
    transform: sitk.Transform,
) -> sitk.Image:
    result = sitk.Resample(
        moving_mask,
        fixed_ct,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    return binarise(result)


def calculate_dice(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
) -> float:
    overlap = sitk.LabelOverlapMeasuresImageFilter()

    overlap.Execute(
        binarise(fixed_mask),
        binarise(moving_mask),
    )

    return float(overlap.GetDiceCoefficient())


def get_centroid(mask: sitk.Image) -> np.ndarray:
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(binarise(mask))

    if not shape.HasLabel(1):
        raise ValueError("Liver mask is empty.")

    return np.asarray(
        shape.GetCentroid(1),
        dtype=float,
    )


def centroid_distance(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
) -> float:
    return float(
        np.linalg.norm(
            get_centroid(fixed_mask)
            - get_centroid(moving_mask)
        )
    )


# ==================================================
# REGISTRATION STAGE 1:
# LIVER STRUCTURE RIGID ALIGNMENT
# ==================================================

def run_structure_rigid_stage(
    fixed_structure: sitk.Image,
    moving_structure: sitk.Image,
    fixed_metric_mask: sitk.Image,
    moving_metric_mask: sitk.Image,
    rigid_transform: sitk.Euler3DTransform,
    label: str,
) -> None:

    print(f"\n{label}: liver-structure rigid stage")

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMeanSquares()

    registration.SetMetricFixedMask(
        fixed_metric_mask
    )

    registration.SetMetricMovingMask(
        moving_metric_mask
    )

    registration.SetMetricSamplingStrategy(
        registration.REGULAR
    )

    registration.SetMetricSamplingPercentage(0.50)

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=0.001,
        numberOfIterations=300,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel(
        shrinkFactors=[4, 2, 1]
    )

    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[2, 1, 0]
    )

    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # inPlace=True means this exact Euler transform is updated.
    registration.SetInitialTransform(
        rigid_transform,
        inPlace=True,
    )

    registration.Execute(
        fixed_structure,
        moving_structure,
    )

    print(
        "  Metric:",
        registration.GetMetricValue(),
    )

    print(
        "  Stop condition:",
        registration.GetOptimizerStopConditionDescription(),
    )


# ==================================================
# REGISTRATION STAGE 2:
# LOCAL CT-INTENSITY REFINEMENT
# ==================================================

def run_ct_refinement_stage(
    fixed_ct: sitk.Image,
    moving_ct: sitk.Image,
    fixed_metric_mask: sitk.Image,
    moving_metric_mask: sitk.Image,
    rigid_transform: sitk.Euler3DTransform,
    label: str,
) -> None:

    print(f"\n{label}: liver-focused CT refinement")

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration.SetMetricFixedMask(
        fixed_metric_mask
    )

    registration.SetMetricMovingMask(
        moving_metric_mask
    )

    registration.SetMetricSamplingStrategy(
        registration.REGULAR
    )

    registration.SetMetricSamplingPercentage(0.30)

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.0005,
        numberOfIterations=300,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel(
        shrinkFactors=[4, 2, 1]
    )

    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[2, 1, 0]
    )

    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(
        rigid_transform,
        inPlace=True,
    )

    registration.Execute(
        clamp_ct(fixed_ct),
        clamp_ct(moving_ct),
    )

    print(
        "  Metric:",
        registration.GetMetricValue(),
    )

    print(
        "  Stop condition:",
        registration.GetOptimizerStopConditionDescription(),
    )


# ==================================================
# COMPLETE REGISTRATION FOR ONE TIMEPOINT
# ==================================================

def register_timepoint(
    *,
    timepoint: str,
    fixed_ct: sitk.Image,
    moving_ct: sitk.Image,
    fixed_liver: sitk.Image,
    moving_liver: sitk.Image,
) -> dict[str, float]:

    print("\n" + "=" * 70)
    print(f"{timepoint} → RTPLAN LIVER-FOCUSED RIGID REGISTRATION")
    print("=" * 70)

    check_mask_matches_ct(
        fixed_liver,
        fixed_ct,
        "RTPLAN",
    )

    check_mask_matches_ct(
        moving_liver,
        moving_ct,
        timepoint,
    )

    fixed_metric_mask = make_metric_mask(
        fixed_liver,
        LIVER_MARGIN_MM,
    )

    moving_metric_mask = make_metric_mask(
        moving_liver,
        LIVER_MARGIN_MM,
    )

    # --------------------------------------------------
    # Initialisation using liver centres of mass
    # --------------------------------------------------

    initial = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_liver, sitk.sitkFloat32),
        sitk.Cast(moving_liver, sitk.sitkFloat32),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    rigid_transform = sitk.Euler3DTransform(
        initial
    )

    initial_liver = resample_mask(
        moving_liver,
        fixed_ct,
        rigid_transform,
    )

    initial_dice = calculate_dice(
        fixed_liver,
        initial_liver,
    )

    print(f"\nInitial liver Dice: {initial_dice:.4f}")

    # --------------------------------------------------
    # Create PlatiPy liver registration structures
    # --------------------------------------------------

    fixed_structure = convert_mask_to_reg_structure(
        fixed_liver,
        expansion=STRUCTURE_EXPANSION,
    )

    moving_structure = convert_mask_to_reg_structure(
        moving_liver,
        expansion=STRUCTURE_EXPANSION,
    )

    fixed_structure = sitk.Cast(
        fixed_structure,
        sitk.sitkFloat32,
    )

    moving_structure = sitk.Cast(
        moving_structure,
        sitk.sitkFloat32,
    )

    # --------------------------------------------------
    # Stage 1: liver shape/boundary alignment
    # --------------------------------------------------

    run_structure_rigid_stage(
        fixed_structure=fixed_structure,
        moving_structure=moving_structure,
        fixed_metric_mask=fixed_metric_mask,
        moving_metric_mask=moving_metric_mask,
        rigid_transform=rigid_transform,
        label=timepoint,
    )

    structure_transform = sitk.Euler3DTransform(
        rigid_transform
    )

    structure_ct = resample_ct(
        moving_ct,
        fixed_ct,
        structure_transform,
    )

    structure_liver = resample_mask(
        moving_liver,
        fixed_ct,
        structure_transform,
    )

    structure_dice = calculate_dice(
        fixed_liver,
        structure_liver,
    )

    print(
        f"Structure-stage liver Dice: "
        f"{structure_dice:.4f}"
    )

    sitk.WriteImage(
        structure_ct,
        str(
            output_dir
            / f"{timepoint}_CT_to_RTPLAN_structure_rigid.nii.gz"
        ),
    )

    sitk.WriteImage(
        structure_liver,
        str(
            output_dir
            / f"{timepoint}_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),
    )

    sitk.WriteTransform(
        structure_transform,
        str(
            output_dir
            / f"{timepoint}_to_RTPLAN_structure_rigid.tfm"
        ),
    )

    # --------------------------------------------------
    # Stage 2: local CT intensity refinement
    # --------------------------------------------------

    run_ct_refinement_stage(
        fixed_ct=fixed_ct,
        moving_ct=moving_ct,
        fixed_metric_mask=fixed_metric_mask,
        moving_metric_mask=moving_metric_mask,
        rigid_transform=rigid_transform,
        label=timepoint,
    )

    final_ct = resample_ct(
        moving_ct,
        fixed_ct,
        rigid_transform,
    )

    final_liver = resample_mask(
        moving_liver,
        fixed_ct,
        rigid_transform,
    )

    final_dice = calculate_dice(
        fixed_liver,
        final_liver,
    )

    final_centroid_distance = centroid_distance(
        fixed_liver,
        final_liver,
    )

    print(f"\nFinal liver Dice: {final_dice:.4f}")
    print(
        f"Final centroid distance: "
        f"{final_centroid_distance:.2f} mm"
    )

    # --------------------------------------------------
    # Save final outputs
    # --------------------------------------------------

    final_ct_path = (
        output_dir
        / f"{timepoint}_CT_to_RTPLAN_liver_focused_rigid.nii.gz"
    )

    final_liver_path = (
        output_dir
        / f"{timepoint}_liver_to_RTPLAN_liver_focused_rigid.nii.gz"
    )

    final_transform_path = (
        output_dir
        / f"{timepoint}_to_RTPLAN_liver_focused_rigid.tfm"
    )

    sitk.WriteImage(
        final_ct,
        str(final_ct_path),
    )

    sitk.WriteImage(
        final_liver,
        str(final_liver_path),
    )

    sitk.WriteTransform(
        rigid_transform,
        str(final_transform_path),
    )

    print("\nSaved:")
    print(f"  CT:        {final_ct_path}")
    print(f"  Liver:     {final_liver_path}")
    print(f"  Transform: {final_transform_path}")

    return {
        "Patient_ID": patient_id,
        "Timepoint": timepoint,
        "Initial_Dice": initial_dice,
        "Structure_Rigid_Dice": structure_dice,
        "Final_Dice": final_dice,
        "Final_Centroid_Distance_mm": final_centroid_distance,
    }


# ==================================================
# MAIN
# ==================================================

def main() -> None:

    required_files = {
        "RTPLAN CT": rtplan_ct_path,
        "RTPLAN liver": rtplan_liver_path,
        "PRE CT": pre_ct_path,
        "PRE liver": pre_liver_path,
        "POST CT": post_ct_path,
        "POST liver": post_liver_path,
    }

    for description, path in required_files.items():
        require_file(path, description)

    fixed_ct = sitk.ReadImage(
        str(rtplan_ct_path),
        sitk.sitkFloat32,
    )

    fixed_liver = binarise(
        sitk.ReadImage(str(rtplan_liver_path))
    )

    cases = {
        "PRE": {
            "ct": sitk.ReadImage(
                str(pre_ct_path),
                sitk.sitkFloat32,
            ),
            "liver": binarise(
                sitk.ReadImage(str(pre_liver_path))
            ),
        },
        "POST": {
            "ct": sitk.ReadImage(
                str(post_ct_path),
                sitk.sitkFloat32,
            ),
            "liver": binarise(
                sitk.ReadImage(str(post_liver_path))
            ),
        },
    }

    qc_results = []

    for timepoint, data in cases.items():
        result = register_timepoint(
            timepoint=timepoint,
            fixed_ct=fixed_ct,
            moving_ct=data["ct"],
            fixed_liver=fixed_liver,
            moving_liver=data["liver"],
        )

        qc_results.append(result)

    qc_path = (
        output_dir
        / f"{patient_id}_liver_focused_rigid_QC.csv"
    )

    with qc_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=qc_results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(qc_results)

    print("\nRegistration complete.")
    print(f"QC results saved to:\n{qc_path}")


if __name__ == "__main__":
    main()
