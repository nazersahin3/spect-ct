from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


# ============================================================
# 1. Patient ID
# ============================================================

PATIENT_ID = "002"


# ------------------------------------------------------------
# DATAFRAME FROM SCRIPT #1
# ------------------------------------------------------------

DATAFRAME_FILE = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/Analysis/evaluation_dataframe_{PATIENT_ID}.csv"
)


# ------------------------------------------------------------
# RTPLAN CT
#
# This is the FIXED / reference image.
# Everything will ultimately be placed onto this grid.
# ------------------------------------------------------------

RTPLAN_CT = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/registered_RTSTRUCT/RTPLAN_CT_converted_from_DICOM copy.nii.gz"
)


# ------------------------------------------------------------
# RAW SPECT IMAGES
#
# CHANGE THESE TWO PATHS
# ------------------------------------------------------------

PRE_SPECT = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-002/SPECT_nifti_pre/Biliary Scan/NM Mebrofenin SPECT F3D AC/1000_Mebrofenin_SPECT_F3D_-_AC.nii.gz"
)

POST_SPECT = Path(
    "/Users/nana/Desktop/spect-data/PRISM-WM-002/SPECT_nifti_post/Biliary Scan/NM Mebrofenin SPECT F3D AC/1000_Mebrofenin_SPECT_F3D_-_AC.nii.gz"
)


# ------------------------------------------------------------
# REGISTRATION TRANSFORMS
#
# These are the transforms you already generated when
# registering PRE and POST to the RTPLAN CT.
#
# CHANGE THESE PATHS.
# ------------------------------------------------------------

PRE_RIGID_TRANSFORM = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/registered_RTSTRUCT/PRE_to_RTPLAN_structure_rigid.tfm"
)

PRE_BSPLINE_TRANSFORM = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/registered_RTSTRUCT/structure_guided_deformable/PRE_to_RTPLAN_boundary_guided_bspline_transform.tfm"
)


POST_RIGID_TRANSFORM = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/registered_RTSTRUCT/POST_to_RTPLAN_structure_rigid.tfm"
)

POST_BSPLINE_TRANSFORM = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/registered_RTSTRUCT/structure_guided_deformable/POST_to_RTPLAN_boundary_guided_bspline_transform.tfm"
)


# ------------------------------------------------------------
# WHOLE-LIVER MASKS
#
# These should already be in RTPLAN space.
#
# They are used for:
#   - whole-liver SPECT sum
#   - TLF scaling
#   - restricting dose bands to liver
#
# CHANGE THESE PATHS.
# ------------------------------------------------------------

PRE_LIVER_MASK = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/registered_RTSTRUCT/PRE_liver_to_RTPLAN_structure_rigid.nii.gz"
)

POST_LIVER_MASK = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/registered_RTSTRUCT/POST_liver_to_RTPLAN_structure_rigid.nii.gz"
)


# ------------------------------------------------------------
# PATIENT TLF VALUES
#
# Enter the actual TLF values for this patient.
# Units = %/min
# ------------------------------------------------------------

PRE_TLF = 12.7       # <-- CHANGE
POST_TLF = 9.9    # <-- CHANGE


# ------------------------------------------------------------
# COUINAUD SEGMENTS -- this is the RT PLAN couinaud segmentation
# ------------------------------------------------------------

COUINAUD_FOLDER = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/{PATIENT_ID}/rt_mask/liver_RT_{PATIENT_ID}_segments_raw_totalsegmentator"
)


# ------------------------------------------------------------
# DOSE BAND LABEL MAP
# ------------------------------------------------------------

DOSE_LABELMAP = Path(
    f"/Users/nana/Desktop/HONOURS/spect-data-imac/NE ALTERED/{PATIENT_ID}/DOSE_CONTOURS/dose_bands_5_to_30Gy.nii.gz"
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_FOLDER = Path(
    f"/Users/nana/Desktop/HONOURS/spect-ct-imac/"
    f"NE ALTERED/{PATIENT_ID}/evaluation"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DATAFRAME = (
    OUTPUT_FOLDER
    / "evaluation_dataframe_evaluated.csv"
)


# ============================================================
# 2. DOSE LABEL DEFINITIONS
# ============================================================

# These correspond to the label map we created previously.

DOSE_LABELS = {
    "<5 Gy": 0,
    "5-10 Gy": 1,
    "10-15 Gy": 2,
    "15-20 Gy": 3,
    "20-25 Gy": 4,
    "25-30 Gy": 5,
    ">=30 Gy": 6,
}


# ============================================================
# 3. CHECK IMAGE GEOMETRY
# ============================================================

def same_geometry(image1, image2, tolerance=1e-5):
    """
    Check whether two images occupy the same image grid.
    """

    return (
        image1.GetSize() == image2.GetSize()
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


# ============================================================
# 4. PUT A MASK ONTO THE RTPLAN GRID
# ============================================================

def prepare_mask(mask, reference):
    """
    Resample a segmentation onto the reference grid if needed.

    Nearest-neighbour interpolation is used because the image
    contains labels rather than continuous intensity data.
    """

    if not same_geometry(mask, reference):

        print(
            "Mask grid differs from RTPLAN grid "
            "-- resampling mask."
        )

        mask = sitk.Resample(
            mask,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            mask.GetPixelID()
        )

    return mask


# ============================================================
# 5. APPLY REGISTRATION TO SPECT
# ============================================================

def register_spect(
    spect_path,
    reference,
    rigid_transform_path,
    bspline_transform_path,
    name
):
    """
    Apply:
        raw SPECT
            -> rigid transform
            -> B-spline transform
            -> RTPLAN space

    This assumes your B-spline registration was performed
    after the rigid registration.
    """

    print()
    print("------------------------------------------")
    print(f"Registering {name} SPECT")
    print("------------------------------------------")

    # Read original SPECT
    spect = sitk.ReadImage(
        str(spect_path),
        sitk.sitkFloat32
    )

    # --------------------------------------------------------
    # RIGID
    # --------------------------------------------------------

    rigid_transform = sitk.ReadTransform(
        str(rigid_transform_path)
    )

    spect_rigid = sitk.Resample(
        spect,
        reference,
        rigid_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )

    print("Rigid transform applied.")

    # --------------------------------------------------------
    # B-SPLINE
    # --------------------------------------------------------

    bspline_transform = sitk.ReadTransform(
        str(bspline_transform_path)
    )

    spect_registered = sitk.Resample(
        spect_rigid,
        reference,
        bspline_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )

    print("B-spline transform applied.")

    # --------------------------------------------------------
    # SAVE REGISTERED SPECT
    # --------------------------------------------------------

    output_path = (
        OUTPUT_FOLDER
        / f"{name}_SPECT_registered_to_RTPLAN.nii.gz"
    )

    sitk.WriteImage(
        spect_registered,
        str(output_path)
    )

    print(
        f"Registered SPECT saved:\n{output_path}"
    )

    return spect_registered


# ============================================================
# 6. TLF SCALE THE REGISTERED SPECT
# ============================================================

def scale_spect_to_tlf(
    spect_image,
    liver_mask,
    patient_tlf,
    name
):
    """
    Scale the SPECT distribution so that the sum of the
    functional values inside the whole liver equals the
    patient's measured TLF.

    scale factor =
        Patient TLF / whole-liver SPECT sum
    """

    print()
    print("------------------------------------------")
    print(f"TLF scaling {name}")
    print("------------------------------------------")

    if patient_tlf <= 0:
        raise ValueError(
            f"{name}_TLF must be greater than zero. "
            "Please enter the patient's actual TLF."
        )

    # Make sure liver mask is on SPECT/reference grid
    liver_mask = prepare_mask(
        liver_mask,
        spect_image
    )

    spect_array = sitk.GetArrayFromImage(
        spect_image
    )

    liver_array = (
        sitk.GetArrayFromImage(
            liver_mask
        ) > 0
    )

    # Raw whole-liver SPECT sum
    whole_liver_sum_counts = float(
        np.sum(
            spect_array[liver_array]
        )
    )

    if whole_liver_sum_counts <= 0:
        raise ValueError(
            f"{name} whole-liver SPECT sum "
            "is zero or negative."
        )

    # --------------------------------------------------------
    # TLF SCALE FACTOR
    # --------------------------------------------------------

    scale_factor = (
        patient_tlf
        / whole_liver_sum_counts
    )

    # Apply scaling to every SPECT voxel
    scaled_array = (
        spect_array
        * scale_factor
    )

    scaled_image = sitk.GetImageFromArray(
        scaled_array.astype(
            np.float32
        )
    )

    scaled_image.CopyInformation(
        spect_image
    )

    # --------------------------------------------------------
    # SAVE TLF-SCALED SPECT
    # --------------------------------------------------------

    output_path = (
        OUTPUT_FOLDER
        / f"{name}_SPECT_TLF_scaled.nii.gz"
    )

    sitk.WriteImage(
        scaled_image,
        str(output_path)
    )

    print(
        f"Whole-liver raw SPECT sum: "
        f"{whole_liver_sum_counts:.3f}"
    )

    print(
        f"Patient TLF: "
        f"{patient_tlf:.4f} %/min"
    )

    print(
        f"TLF scale factor: "
        f"{scale_factor:.10f}"
    )

    print(
        f"Scaled SPECT saved:\n{output_path}"
    )

    return (
        scaled_image,
        whole_liver_sum_counts,
        scale_factor,
        liver_mask
    )


# ============================================================
# 7. CALCULATE SEGMENT STATISTICS
# ============================================================

def calculate_statistics(
    raw_spect,
    scaled_spect,
    mask,
    whole_liver_sum_counts,
    patient_tlf
):
    """
    Calculate statistics within one ROI.

    Raw registered SPECT is used for:
        ROI sum counts

    TLF-scaled SPECT is used for:
        minimum
        maximum
        mean
        SD
        percentiles
        median
        ROI TLF
    """

    # Make sure mask is on same grid
    mask = prepare_mask(
        mask,
        scaled_spect
    )

    raw_array = sitk.GetArrayFromImage(
        raw_spect
    )

    scaled_array = sitk.GetArrayFromImage(
        scaled_spect
    )

    mask_array = (
        sitk.GetArrayFromImage(mask) > 0
    )

    voxel_count = int(
        np.sum(mask_array)
    )

    if voxel_count == 0:
        return None

    # Raw counts inside ROI
    raw_values = raw_array[
        mask_array
    ]

    # TLF-scaled values inside ROI
    scaled_values = scaled_array[
        mask_array
    ]

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    spacing = scaled_spect.GetSpacing()

    voxel_volume_mm3 = (
        spacing[0]
        * spacing[1]
        * spacing[2]
    )

    voxel_volume_cm3 = (
        voxel_volume_mm3
        / 1000.0
    )

    volume_cm3 = (
        voxel_count
        * voxel_volume_cm3
    )

    # --------------------------------------------------------
    # RAW ROI COUNTS
    # --------------------------------------------------------

    roi_sum_counts = float(
        np.sum(raw_values)
    )

    # Fraction of total liver counts
    roi_fraction_liver_counts = (
        roi_sum_counts
        / whole_liver_sum_counts
    )

    # --------------------------------------------------------
    # REGIONAL TLF
    # --------------------------------------------------------

    roi_tlf = (
        roi_fraction_liver_counts
        * patient_tlf
    )

    # This should give essentially the same result:
    roi_tlf_from_scaled_image = float(
        np.sum(scaled_values)
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    statistics = {

        "volume_cm3":
            volume_cm3,

        "roi_sum_counts":
            roi_sum_counts,

        "whole_liver_sum_counts":
            whole_liver_sum_counts,

        "roi_fraction_liver_counts":
            roi_fraction_liver_counts,

        "patient_tlf_pct_min":
            patient_tlf,

        "roi_tlf_pct_min":
            roi_tlf,

        "minimum":
            float(np.min(scaled_values)),

        "maximum":
            float(np.max(scaled_values)),

        "mean":
            float(np.mean(scaled_values)),

        "standard_deviation":
            float(np.std(scaled_values)),

        "percentile_5":
            float(
                np.percentile(
                    scaled_values,
                    5
                )
            ),

        "percentile_95":
            float(
                np.percentile(
                    scaled_values,
                    95
                )
            ),

        "median":
            float(np.median(scaled_values)),
    }

    # Small QC check
    difference = abs(
        roi_tlf
        - roi_tlf_from_scaled_image
    )

    if difference > 1e-5:
        print(
            "WARNING: ROI TLF calculations "
            "do not perfectly agree."
        )

    return statistics


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print(f"PATIENT {PATIENT_ID}")
    print("==========================================")


    # ========================================================
    # LOAD DATAFRAME
    # ========================================================

    df = pd.read_csv(
        DATAFRAME_FILE,
        dtype={
            "patient_id": str
        }
    )

    # Preserve leading zeros
    df["patient_id"] = (
        df["patient_id"]
        .str.zfill(3)
    )


    # ========================================================
    # ADD STATISTICS COLUMNS IF THEY DO NOT EXIST
    # ========================================================

    statistic_columns = [

        "volume_cm3",

        "roi_sum_counts",
        "whole_liver_sum_counts",
        "roi_fraction_liver_counts",

        "patient_tlf_pct_min",
        "roi_tlf_pct_min",

        "minimum",
        "maximum",
        "mean",
        "standard_deviation",

        "percentile_5",
        "percentile_95",
        "median",
    ]

    for column in statistic_columns:

        if column not in df.columns:
            df[column] = np.nan


    # ========================================================
    # LOAD RTPLAN REFERENCE
    # ========================================================

    reference = sitk.ReadImage(
        str(RTPLAN_CT)
    )


    # ========================================================
    # REGISTER PRE SPECT
    # ========================================================

    pre_registered = register_spect(

        spect_path=PRE_SPECT,

        reference=reference,

        rigid_transform_path=
            PRE_RIGID_TRANSFORM,

        bspline_transform_path=
            PRE_BSPLINE_TRANSFORM,

        name="PRE"
    )


    # ========================================================
    # REGISTER POST SPECT
    # ========================================================

    post_registered = register_spect(

        spect_path=POST_SPECT,

        reference=reference,

        rigid_transform_path=
            POST_RIGID_TRANSFORM,

        bspline_transform_path=
            POST_BSPLINE_TRANSFORM,

        name="POST"
    )


    # ========================================================
    # LOAD WHOLE-LIVER MASKS
    # ========================================================

    pre_liver = sitk.ReadImage(
        str(PRE_LIVER_MASK)
    )

    post_liver = sitk.ReadImage(
        str(POST_LIVER_MASK)
    )


    # ========================================================
    # TLF SCALE PRE
    # ========================================================

    (
        pre_scaled,
        pre_whole_liver_sum,
        pre_scale_factor,
        pre_liver
    ) = scale_spect_to_tlf(

        spect_image=
            pre_registered,

        liver_mask=
            pre_liver,

        patient_tlf=
            PRE_TLF,

        name="PRE"
    )


    # ========================================================
    # TLF SCALE POST
    # ========================================================

    (
        post_scaled,
        post_whole_liver_sum,
        post_scale_factor,
        post_liver
    ) = scale_spect_to_tlf(

        spect_image=
            post_registered,

        liver_mask=
            post_liver,

        patient_tlf=
            POST_TLF,

        name="POST"
    )


    # ========================================================
    # LOAD DOSE LABEL MAP
    # ========================================================

    dose_image = sitk.ReadImage(
        str(DOSE_LABELMAP)
    )

    # Put dose labels onto RTPLAN grid if needed
    dose_image = prepare_mask(
        dose_image,
        reference
    )


    # ========================================================
    # GO THROUGH DATAFRAME ROWS
    # ========================================================

    print()
    print("==========================================")
    print("CALCULATING SEGMENT STATISTICS")
    print("==========================================")


    for index, row in df.iterrows():


        # ----------------------------------------------------
        # Only process current patient
        # ----------------------------------------------------

        if (
            row["patient_id"]
            != PATIENT_ID
        ):
            continue


        # ----------------------------------------------------
        # CHOOSE TIMEPOINT
        # ----------------------------------------------------

        if row["timepoint"] == "PRE":

            raw_spect = pre_registered

            scaled_spect = pre_scaled

            liver_mask = pre_liver

            whole_liver_sum = (
                pre_whole_liver_sum
            )

            patient_tlf = PRE_TLF


        elif row["timepoint"] == "POST":

            raw_spect = post_registered

            scaled_spect = post_scaled

            liver_mask = post_liver

            whole_liver_sum = (
                post_whole_liver_sum
            )

            patient_tlf = POST_TLF


        else:
            continue


        # ====================================================
        # COUINAUD ROI
        # ====================================================

        if (
            row["roi_type"]
            == "Couinaud"
        ):

            segment_number = int(

                row["roi_name"]
                .replace(
                    "Segment ",
                    ""
                )
            )

            mask_path = (

                COUINAUD_FOLDER

                / (
                    f"liver_segment_"
                    f"{segment_number}.nii.gz"
                )
            )

            mask = sitk.ReadImage(
                str(mask_path)
            )

            mask = prepare_mask(
                mask,
                reference
            )


        # ====================================================
        # DOSE ROI
        # ====================================================

        elif (
            row["roi_type"]
            == "Dose"
        ):

            roi_name = (
                row["roi_name"]
            )

            label_value = (
                DOSE_LABELS[
                    roi_name
                ]
            )

            # Turn one dose label into
            # a binary mask
            mask = sitk.Cast(

                dose_image
                == label_value,

                sitk.sitkUInt8
            )


            # ------------------------------------------------
            # RESTRICT DOSE BAND TO LIVER
            #
            # This is important because the original dose
            # label map also contains voxels outside liver.
            # ------------------------------------------------

            liver_binary = sitk.Cast(
                liver_mask > 0,
                sitk.sitkUInt8
            )

            mask = sitk.And(
                mask,
                liver_binary
            )


        else:
            continue


        # ====================================================
        # CALCULATE STATISTICS
        # ====================================================

        stats = calculate_statistics(

            raw_spect=
                raw_spect,

            scaled_spect=
                scaled_spect,

            mask=
                mask,

            whole_liver_sum_counts=
                whole_liver_sum,

            patient_tlf=
                patient_tlf
        )


        if stats is None:

            print(
                f"No voxels found: "
                f"{row['timepoint']} - "
                f"{row['roi_name']}"
            )

            continue


        # ====================================================
        # PUT VALUES INTO DATAFRAME
        # ====================================================

        for column, value in stats.items():

            df.loc[
                index,
                column
            ] = value


        print(
            f"Finished: "
            f"{row['timepoint']} | "
            f"{row['roi_type']} | "
            f"{row['roi_name']}"
        )


    # ========================================================
    # SAVE COMPLETED DATAFRAME
    # ========================================================

    df.to_csv(
        OUTPUT_DATAFRAME,
        index=False
    )


    print()
    print("==========================================")
    print("FINISHED")
    print("==========================================")

    print(
        f"Completed dataframe saved to:\n"
        f"{OUTPUT_DATAFRAME}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()