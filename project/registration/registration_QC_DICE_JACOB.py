from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# 1. OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = Path(
    "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
    "Analysis/registration_QC"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. EXPLICIT PATIENT PATHS
# ============================================================

PATIENT_PATHS = {


    # ========================================================
    # PATIENT 002
    # ========================================================

    "002": {

        "rtplan_ct": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/002/registered_RTSTRUCT/RTPLAN_CT_converted_from_DICOM copy.nii.gz"
        ),

        "rtplan_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/002/rtplan_structures/RTPLAN_Liver.nii.gz"
        ),


        # ----------------------------------------------------
        # ORIGINAL / UNREGISTERED LIVER MASKS
        #
        # Put the real paths here when you locate them.
        # ----------------------------------------------------

        "PRE_original_liver": Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/002/liver_segmentation_pre_002/liver_mask_002_pre.nii"),

        "POST_original_liver": Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/002/liver_segmentation_post_002/liver_post002_mask.nii.gz"),


        # ----------------------------------------------------
        # AFTER RIGID REGISTRATION
        # ----------------------------------------------------

        "PRE_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/002/registered_RTSTRUCT/"
            "PRE_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),

        "POST_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/002/registered_RTSTRUCT/"
            "POST_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),


        # ----------------------------------------------------
        # AFTER DEFORMABLE REGISTRATION
        # ----------------------------------------------------

        "PRE_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/002/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),

        "POST_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/002/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),


        # ----------------------------------------------------
        # B-SPLINE TRANSFORMS
        # ----------------------------------------------------

        "PRE_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/002/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),

        "POST_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/002/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),
    },


    # ========================================================
    # PATIENT 004
    # ========================================================

    "004": {

        "rtplan_ct": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/004/registered_RTSTRUCT/"
            "RTPLAN_CT_converted_from_DICOM copy.nii.gz"
        ),

        "rtplan_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-data-imac/"
            "NE ALTERED/004/rtplan_structures/"
            "RTPLAN_Liver.nii.gz"
        ),

        "PRE_original_liver": Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/liver_segmentation_pre004/liver_004_pre_mask.nii.gz"),
        "POST_original_liver": Path("/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/liver_segmentation_post004/liver_post004_mask.nii.gz"),

        "PRE_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/004/registered_RTSTRUCT/"
            "PRE_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),

        "POST_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/004/registered_RTSTRUCT/"
            "POST_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),

        "PRE_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/004/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),

        "POST_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/004/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),

        "PRE_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/004/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),

        "POST_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/004/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),
    },


    # ========================================================
    # PATIENT 005
    # ========================================================

    "005": {

        "rtplan_ct": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/registered/"
            "RTPLAN_CT_converted_from_DICOM.nii.gz"
        ),

        "rtplan_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/005/rtplan_structures/RTPLAN_Liver.nii.gz"
        ),

        "PRE_original_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/liver_segmentation_pre_005/"
            "liver_pre_005_mask.nii.gz"
        ),

        "POST_original_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/liver_segmentation_post_005/"
            "liver_post_005_mask.nii.gz"
        ),

        "PRE_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/registered_RTSTRUCT/"
            "PRE_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),

        "POST_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/registered_RTSTRUCT/"
            "POST_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),

        "PRE_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),

        "POST_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),

        "PRE_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),

        "POST_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/005/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),
    },


    # ========================================================
    # PATIENT 006
    # ========================================================

    "006": {

        "rtplan_ct": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/registered/"
            "RTPLAN_CT_converted_from_DICOM.nii.gz"
        ),

        "rtplan_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-data-imac/"
            "NE ALTERED/006/rtplan_structures/"
            "RTPLAN_Liver.nii.gz"
        ),

        "PRE_original_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/liver_segmentation_pre_006/"
            "liver_pre_006_mask.nii.gz"
        ),

        "POST_original_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/liver_segmentation_post_006/"
            "liver_post_006_mask.nii.gz"
        ),

        "PRE_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/registered_RTSTRUCT/"
            "PRE_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),

        "POST_rigid_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/registered_RTSTRUCT/"
            "POST_liver_to_RTPLAN_structure_rigid.nii.gz"
        ),

        "PRE_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),

        "POST_deformable_liver": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_liver_to_RTPLAN_boundary_guided_bspline.nii.gz"
        ),

        "PRE_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "PRE_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),

        "POST_bspline_transform": Path(
            "/Users/nana/Desktop/HONOURS/spect-ct-imac/"
            "NE ALTERED/006/registered_RTSTRUCT/"
            "structure_guided_deformable/"
            "POST_to_RTPLAN_boundary_guided_bspline_transform.tfm"
        ),
    },
}


# ============================================================
# 3. BASIC FUNCTIONS
# ============================================================

def binarise(image):

    return sitk.Cast(
        image > 0,
        sitk.sitkUInt8
    )


def require_file(
    path,
    description
):

    if path is None:

        return False

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    return True


def same_geometry(
    image1,
    image2,
    tolerance=1e-5
):

    return (
        image1.GetSize()
        == image2.GetSize()

        and np.allclose(
            image1.GetSpacing(),
            image2.GetSpacing(),
            atol=tolerance
        )

        and np.allclose(
            image1.GetOrigin(),
            image2.GetOrigin(),
            atol=tolerance
        )

        and np.allclose(
            image1.GetDirection(),
            image2.GetDirection(),
            atol=tolerance
        )
    )


def put_mask_on_reference(
    mask,
    reference
):

    if same_geometry(
        mask,
        reference
    ):

        return binarise(mask)

    return sitk.Resample(
        binarise(mask),
        reference,
        sitk.Transform(
            3,
            sitk.sitkIdentity
        ),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )


# ============================================================
# 4. DICE
# ============================================================

def calculate_dice(
    fixed_mask,
    moving_mask
):

    fixed = (
        sitk.GetArrayFromImage(
            fixed_mask
        ) > 0
    )

    moving = (
        sitk.GetArrayFromImage(
            moving_mask
        ) > 0
    )

    intersection = (
        np.logical_and(
            fixed,
            moving
        ).sum()
    )

    denominator = (
        fixed.sum()
        + moving.sum()
    )

    if denominator == 0:

        return np.nan

    return float(
        2
        * intersection
        / denominator
    )


# ============================================================
# 5. JACOBIAN
# ============================================================

def calculate_jacobian(
    transform_path,
    reference_ct,
    liver_mask,
    patient_id,
    timepoint
):

    transform = sitk.ReadTransform(
        str(transform_path)
    )


    # --------------------------------------------------------
    # B-spline transform -> displacement field
    # --------------------------------------------------------

    displacement_filter = (
        sitk.TransformToDisplacementFieldFilter()
    )

    displacement_filter.SetReferenceImage(
        reference_ct
    )

    displacement_field = (
        displacement_filter.Execute(
            transform
        )
    )


    # --------------------------------------------------------
    # Displacement field -> Jacobian determinant
    # --------------------------------------------------------

    jacobian = (
        sitk.DisplacementFieldJacobianDeterminant(
            displacement_field,
            True
        )
    )


    # --------------------------------------------------------
    # Save full Jacobian image
    # --------------------------------------------------------

    jacobian_path = (
        OUTPUT_DIR
        / (
            f"{patient_id}_{timepoint}_"
            "jacobian.nii.gz"
        )
    )

    sitk.WriteImage(
        jacobian,
        str(jacobian_path),
        useCompression=True
    )


    # --------------------------------------------------------
    # Only evaluate Jacobian inside RTPLAN liver
    # --------------------------------------------------------

    jac_array = (
        sitk.GetArrayFromImage(
            jacobian
        )
    )

    liver_array = (
        sitk.GetArrayFromImage(
            liver_mask
        ) > 0
    )


    values = (
        jac_array[
            liver_array
        ]
    )

    values = values[
        np.isfinite(values)
    ]


    return {

        "jacobian_mean":
            np.mean(values),

        "jacobian_median":
            np.median(values),

        "jacobian_std":
            np.std(values),

        "jacobian_min":
            np.min(values),

        "jacobian_p05":
            np.percentile(
                values,
                5
            ),

        "jacobian_p95":
            np.percentile(
                values,
                95
            ),

        "jacobian_max":
            np.max(values),

        "jacobian_nonpositive_pct":
            100
            * np.mean(
                values <= 0
            ),

        "jacobian_below_1_pct":
            100
            * np.mean(
                (values > 0)
                &
                (values < 1)
            ),

        "jacobian_above_1_pct":
            100
            * np.mean(
                values > 1
            ),
    }


# ============================================================
# 6. PROCESS ONE TIMEPOINT
# ============================================================

def process_timepoint(
    patient_id,
    timepoint,
    paths,
    reference_ct,
    fixed_liver
):

    print()
    print("------------------------------------------")
    print(
        f"Patient {patient_id} | {timepoint}"
    )
    print("------------------------------------------")


    original_path = paths[
        f"{timepoint}_original_liver"
    ]

    rigid_path = paths[
        f"{timepoint}_rigid_liver"
    ]

    deformable_path = paths[
        f"{timepoint}_deformable_liver"
    ]

    transform_path = paths[
        f"{timepoint}_bspline_transform"
    ]


    # --------------------------------------------------------
    # REQUIRED FILES
    # --------------------------------------------------------

    require_file(
        rigid_path,
        f"{timepoint} rigid liver"
    )

    require_file(
        deformable_path,
        f"{timepoint} deformable liver"
    )

    require_file(
        transform_path,
        f"{timepoint} B-spline transform"
    )


    # ========================================================
    # RIGID DICE
    # ========================================================

    rigid_liver = (
        put_mask_on_reference(
            sitk.ReadImage(
                str(rigid_path)
            ),
            reference_ct
        )
    )

    dice_rigid = (
        calculate_dice(
            fixed_liver,
            rigid_liver
        )
    )


    # ========================================================
    # DEFORMABLE DICE
    # ========================================================

    deformable_liver = (
        put_mask_on_reference(
            sitk.ReadImage(
                str(deformable_path)
            ),
            reference_ct
        )
    )

    dice_deformable = (
        calculate_dice(
            fixed_liver,
            deformable_liver
        )
    )


    # ========================================================
    # ORIGINAL / UNREGISTERED DICE
    #
    # Only calculated if a path was supplied.
    # ========================================================

    if original_path is not None:

        require_file(
            original_path,
            f"{timepoint} original liver"
        )

        original_liver = (
            put_mask_on_reference(
                sitk.ReadImage(
                    str(original_path)
                ),
                reference_ct
            )
        )

        dice_original = (
            calculate_dice(
                fixed_liver,
                original_liver
            )
        )

    else:

        dice_original = np.nan


    # ========================================================
    # JACOBIAN
    # ========================================================

    jacobian_stats = (
        calculate_jacobian(
            transform_path=
                transform_path,

            reference_ct=
                reference_ct,

            liver_mask=
                fixed_liver,

            patient_id=
                patient_id,

            timepoint=
                timepoint
        )
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    if np.isfinite(
        dice_original
    ):

        print(
            f"Dice before registration:  "
            f"{dice_original:.4f}"
        )

    else:

        print(
            "Dice before registration:  "
            "not calculated "
            "(original liver path not supplied)"
        )


    print(
        f"Dice after rigid:          "
        f"{dice_rigid:.4f}"
    )

    print(
        f"Dice after deformable:     "
        f"{dice_deformable:.4f}"
    )


    # ========================================================
    # RESULT ROW
    # ========================================================

    result = {

        "patient_id":
            patient_id,

        "timepoint":
            timepoint,

        "dice_before":
            dice_original,

        "dice_after_rigid":
            dice_rigid,

        "dice_after_deformable":
            dice_deformable,

        "dice_rigid_to_deformable_improvement":
            dice_deformable
            - dice_rigid,
    }


    if np.isfinite(
        dice_original
    ):

        result[
            "dice_total_improvement"
        ] = (
            dice_deformable
            - dice_original
        )

    else:

        result[
            "dice_total_improvement"
        ] = np.nan


    result.update(
        jacobian_stats
    )


    return result


# ============================================================
# 7. CHECK ALL EXPLICIT PATHS FIRST
#
# This prints missing paths BEFORE analysis begins.
# ============================================================

print()
print("==========================================")
print("CHECKING INPUT PATHS")
print("==========================================")


missing_files = []


for patient_id, paths in PATIENT_PATHS.items():

    for name, path in paths.items():

        if path is None:

            print(
                f"{patient_id} | "
                f"{name}: "
                "NOT SUPPLIED"
            )

            continue


        if path.exists():

            print(
                f"{patient_id} | "
                f"{name}: OK"
            )

        else:

            print(
                f"{patient_id} | "
                f"{name}: MISSING"
            )

            missing_files.append(
                (
                    patient_id,
                    name,
                    path
                )
            )


# ============================================================
# 8. RUN ANALYSIS
# ============================================================

results = []


for patient_id, paths in PATIENT_PATHS.items():

    print()
    print("==========================================")
    print(
        f"PROCESSING PATIENT {patient_id}"
    )
    print("==========================================")


    require_file(
        paths["rtplan_ct"],
        "RTPLAN CT"
    )

    require_file(
        paths["rtplan_liver"],
        "RTPLAN liver"
    )


    reference_ct = (
        sitk.ReadImage(
            str(
                paths[
                    "rtplan_ct"
                ]
            ),
            sitk.sitkFloat32
        )
    )


    fixed_liver = (
        put_mask_on_reference(
            sitk.ReadImage(
                str(
                    paths[
                        "rtplan_liver"
                    ]
                )
            ),
            reference_ct
        )
    )


    for timepoint in [
        "PRE",
        "POST"
    ]:

        result = (
            process_timepoint(
                patient_id=
                    patient_id,

                timepoint=
                    timepoint,

                paths=
                    paths,

                reference_ct=
                    reference_ct,

                fixed_liver=
                    fixed_liver
            )
        )

        results.append(
            result
        )


# ============================================================
# 9. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


results_csv = (
    OUTPUT_DIR
    / "registration_QC_dice_jacobian.csv"
)


results_df.to_csv(
    results_csv,
    index=False
)


# ============================================================
# 10. THESIS-FRIENDLY TABLE
# ============================================================

table_df = results_df[
    [
        "patient_id",
        "timepoint",
        "dice_before",
        "dice_after_rigid",
        "dice_after_deformable",
        "jacobian_mean",
        "jacobian_p05",
        "jacobian_p95",
        "jacobian_nonpositive_pct",
    ]
].copy()


table_df.columns = [
    "Patient",
    "Timepoint",
    "Dice Before",
    "Dice Rigid",
    "Dice Deformable",
    "Jacobian Mean",
    "Jacobian P5",
    "Jacobian P95",
    "Jacobian <= 0 (%)",
]


table_df = (
    table_df.round(4)
)


print()
print("==========================================")
print("REGISTRATION QC TABLE")
print("==========================================")
print()

print(
    table_df.to_string(
        index=False
    )
)


thesis_csv = (
    OUTPUT_DIR
    / "registration_QC_table.csv"
)


table_df.to_csv(
    thesis_csv,
    index=False
)


# ============================================================
# 11. SAVE TABLE AS IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 4)
)

ax.axis(
    "off"
)


table = ax.table(
    cellText=
        table_df.values,

    colLabels=
        table_df.columns,

    cellLoc=
        "center",

    loc=
        "center"
)


table.auto_set_font_size(
    False
)

table.set_fontsize(
    9
)

table.scale(
    1,
    1.5
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "registration_QC_table.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print()
print("==========================================")
print("FINISHED")
print("==========================================")

print(
    f"Full results:\n"
    f"{results_csv}"
)

print()

print(
    f"Thesis table:\n"
    f"{thesis_csv}"
)