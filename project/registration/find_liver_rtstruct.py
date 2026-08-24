from pathlib import Path
import pydicom


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

patient_id = "002"

# This is the OUTER folder containing all RTSTRUCT folders/files.
rtstruct_root = Path(
    f"/Users/nana/Desktop/spect-data/PRISM-WM-{patient_id}/RT/RTSTRUCTS"
)

# This must be the planning CT DICOM series used for contouring.
planning_ct_dir = Path(
    f"/Users/nana/Desktop/spect-data/PRISM-WM-{patient_id}/RT/CT/DICOM"
)

target_structure_name = "Liver"


# --------------------------------------------------
# READ A DICOM HEADER
# --------------------------------------------------

def read_dicom_header(file_path: Path):
    """
    Read a DICOM header without loading pixel data.

    Returns None when the file is not a readable DICOM object.
    """
    try:
        return pydicom.dcmread(
            str(file_path),
            stop_before_pixels=True,
        )
    except Exception:
        return None


# --------------------------------------------------
# FIND PLANNING CT SERIES UID
# --------------------------------------------------

def get_planning_ct_series_uid(ct_directory: Path) -> str:
    """
    Find the SeriesInstanceUID of the planning CT series.
    """

    if not ct_directory.exists():
        raise FileNotFoundError(
            f"Planning CT folder does not exist:\n{ct_directory}"
        )

    for file_path in ct_directory.rglob("*"):

        if not file_path.is_file():
            continue

        # Ignore macOS metadata files.
        if file_path.name.startswith("._"):
            continue

        ds = read_dicom_header(file_path)

        if ds is None:
            continue

        modality = str(
            getattr(ds, "Modality", "")
        ).upper()

        if modality != "CT":
            continue

        series_uid = getattr(
            ds,
            "SeriesInstanceUID",
            None,
        )

        if series_uid:
            return str(series_uid)

    raise RuntimeError(
        f"No CT DICOM files were found in:\n{ct_directory}"
    )


# --------------------------------------------------
# GET CT SERIES REFERENCED BY AN RTSTRUCT
# --------------------------------------------------

def get_referenced_series_uids(rtstruct_dataset) -> set[str]:
    """
    Extract CT SeriesInstanceUID values referenced by an RTSTRUCT.
    """

    referenced_uids = set()

    frame_sequence = getattr(
        rtstruct_dataset,
        "ReferencedFrameOfReferenceSequence",
        [],
    )

    for frame_item in frame_sequence:

        study_sequence = getattr(
            frame_item,
            "RTReferencedStudySequence",
            [],
        )

        for study_item in study_sequence:

            series_sequence = getattr(
                study_item,
                "RTReferencedSeriesSequence",
                [],
            )

            for series_item in series_sequence:

                uid = getattr(
                    series_item,
                    "SeriesInstanceUID",
                    None,
                )

                if uid:
                    referenced_uids.add(str(uid))

    return referenced_uids


# --------------------------------------------------
# MAIN SEARCH
# --------------------------------------------------

if not rtstruct_root.exists():
    raise FileNotFoundError(
        f"RTSTRUCT folder does not exist:\n{rtstruct_root}"
    )

planning_ct_uid = get_planning_ct_series_uid(
    planning_ct_dir
)

print("\nPlanning CT SeriesInstanceUID:")
print(planning_ct_uid)

print("\nSearching RTSTRUCT folder:")
print(rtstruct_root)

candidates = []


for file_path in rtstruct_root.rglob("*"):

    if not file_path.is_file():
        continue

    # Ignore macOS metadata files.
    if file_path.name.startswith("._"):
        continue

    ds = read_dicom_header(file_path)

    if ds is None:
        continue

    modality = str(
        getattr(ds, "Modality", "")
    ).upper()

    if modality != "RTSTRUCT":
        continue

    roi_entries = [
        {
            "number": int(item.ROINumber),
            "name": str(item.ROIName).strip(),
        }
        for item in getattr(
            ds,
            "StructureSetROISequence",
            [],
        )
    ]

    # Select only the ROI named exactly "Liver".
    exact_liver_matches = [
        roi
        for roi in roi_entries
        if roi["name"].casefold()
        == target_structure_name.casefold()
    ]

    if not exact_liver_matches:
        continue

    referenced_uids = get_referenced_series_uids(ds)

    references_planning_ct = (
        planning_ct_uid in referenced_uids
    )

    structure_set_label = str(
        getattr(
            ds,
            "StructureSetLabel",
            "No label",
        )
    )

    structure_set_date = str(
        getattr(
            ds,
            "StructureSetDate",
            "No date",
        )
    )

    print("\n----------------------------------------")
    print("RTSTRUCT candidate:")
    print(file_path)

    print(
        f"Structure-set label: "
        f"{structure_set_label}"
    )

    print(
        f"Structure-set date:  "
        f"{structure_set_date}"
    )

    print("Exact liver ROI:")

    for liver_roi in exact_liver_matches:
        print(
            f'  Name: {liver_roi["name"]}, '
            f'ROI number: {liver_roi["number"]}'
        )

    print(
        "References selected planning CT:",
        references_planning_ct,
    )

    candidates.append(
        {
            "path": file_path,
            "liver_rois": exact_liver_matches,
            "references_planning_ct":
                references_planning_ct,
            "label": structure_set_label,
            "date": structure_set_date,
        }
    )


# --------------------------------------------------
# SELECT THE BEST CANDIDATE
# --------------------------------------------------

matching_candidates = [
    candidate
    for candidate in candidates
    if candidate["references_planning_ct"]
]


if len(matching_candidates) == 1:

    selected = matching_candidates[0]
    selected_liver = selected["liver_rois"][0]

    print("\n========================================")
    print("USE THIS RTSTRUCT FILE:")
    print(selected["path"])

    print("\nSELECTED STRUCTURE:")
    print(f'  Name:       {selected_liver["name"]}')
    print(f'  ROI number: {selected_liver["number"]}')

    print("\nReferences selected planning CT:")
    print(True)
    print("========================================")


elif len(matching_candidates) > 1:

    print(
        "\nMultiple RTSTRUCT files contain an exact ROI "
        "named 'Liver' and reference the selected planning CT."
    )

    print(
        "Compare the structure-set labels and dates printed "
        "above before choosing one."
    )


elif candidates:

    print(
        "\nOne or more RTSTRUCT files contain an exact ROI "
        "named 'Liver', but none reference the selected "
        "planning CT series."
    )

    print(
        "Check that planning_ct_dir points to the CT series "
        "used when the structures were drawn."
    )


else:

    print(
        "\nNo RTSTRUCT containing an ROI named exactly "
        "'Liver' was found."
    )