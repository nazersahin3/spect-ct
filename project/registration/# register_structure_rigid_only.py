# register_structure_rigid_only.py

from pathlib import Path
import csv

import numpy as np
import SimpleITK as sitk

from platipy.imaging.registration.utils import (
    convert_mask_to_reg_structure,
)


# ==================================================
# USER INPUTS
# ==================================================

patient_id = "004"

base_dir = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED"
)

output_dir = base_dir / patient_id / "registered_RTSTRUCT"
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FIXED RTPLAN CT AND LIVER MASK
# --------------------------------------------------

rtplan_ct_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/registered/RTPLAN_CT_converted_from_DICOM.nii.gz"
    )

# Replace with the TotalSegmentator liver mask with the RTSTRUCT Liver.
rtplan_liver_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/004/rtplan_structures/RTPLAN_Liver.nii.gz" 
)

# --------------------------------------------------
# ORIGINAL PRE AND POST CTs
# --------------------------------------------------

pre_ct_path = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-004/SPECT_nifti_pre/Biliary Scan/CT AC AbdoLowDose 3.0 I41s/003_AC__AbdoLowDose__3_0__I41s.nii.gz"
)

post_ct_path = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-004/SPECT_nifti_post/Biliary Scan/3 CT_Liver_3mm_I41s.nii"
)


# --------------------------------------------------
# ORIGINAL PRE AND POST LIVER MASKS
#
# Each mask must have been generated from the exact
# corresponding CT listed above.
# --------------------------------------------------

pre_liver_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/liver_segmentation_pre004/liver_004_pre_mask.nii.gz"
)

post_liver_path = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/liver_segmentation_post004/liver_post004_mask.nii.gz"
)


# ==================================================
# SETTINGS
# ==================================================

# Amount of surrounding anatomy included in the
# structure-registration metric masks.
LIVER_MARGIN_MM = 30.0

# Slight mask expansion for smoother structure images.
STRUCTURE_EXPANSION = (3, 3, 3)


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
# LIVER-STRUCTURE RIGID REGISTRATION
# ==================================================

def run_structure_rigid_stage(
    fixed_structure: sitk.Image,
    moving_structure: sitk.Image,
    fixed_metric_mask: sitk.Image,
    moving_metric_mask: sitk.Image,
    rigid_transform: sitk.Euler3DTransform,
    label: str,
) -> None:

    print(f"\n{label}: liver-structure rigid registration")

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

    # inPlace=True updates this Euler transform directly.
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
    print(f"{timepoint} → RTPLAN STRUCTURE-BASED RIGID REGISTRATION")
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
    # Structure-based rigid registration only
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

    structure_centroid_distance = centroid_distance(
        fixed_liver,
        structure_liver,
    )

    print(
        f"\nStructure-stage liver Dice: "
        f"{structure_dice:.4f}"
    )

    print(
        f"Structure-stage centroid distance: "
        f"{structure_centroid_distance:.2f} mm"
    )

    # --------------------------------------------------
    # Save structure-registration outputs
    # --------------------------------------------------

    ct_output_path = (
        output_dir
        / f"{timepoint}_CT_to_RTPLAN_structure_rigid.nii.gz"
    )

    liver_output_path = (
        output_dir
        / f"{timepoint}_liver_to_RTPLAN_structure_rigid.nii.gz"
    )

    transform_output_path = (
        output_dir
        / f"{timepoint}_to_RTPLAN_structure_rigid.tfm"
    )

    sitk.WriteImage(
        structure_ct,
        str(ct_output_path),
    )

    sitk.WriteImage(
        structure_liver,
        str(liver_output_path),
    )

    sitk.WriteTransform(
        structure_transform,
        str(transform_output_path),
    )

    print("\nSaved:")
    print(f"  CT:        {ct_output_path}")
    print(f"  Liver:     {liver_output_path}")
    print(f"  Transform: {transform_output_path}")

    return {
        "Patient_ID": patient_id,
        "Timepoint": timepoint,
        "Initial_Dice": initial_dice,
        "Structure_Rigid_Dice": structure_dice,
        "Structure_Centroid_Distance_mm":
            structure_centroid_distance,
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
        / f"{patient_id}_structure_rigid_QC.csv"
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