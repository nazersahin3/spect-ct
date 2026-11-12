from multiprocessing import freeze_support
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator


def main():
    case_id = "RT_004"

    input_ct = "/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/registered_RTSTRUCT/RTPLAN_CT_converted_from_DICOM.nii.gz"

    out_dir = Path(f"/Users/nana/Desktop/HONOURS/spect-ct-imac/NE ALTERED/004/rt_mask")
    out_dir.mkdir(parents=True, exist_ok=True)

    segments_dir = out_dir / f"liver_{case_id}_segments_raw_totalsegmentator"
    segments_dir.mkdir(parents=True, exist_ok=True)

    totalsegmentator(
        input=input_ct,
        output=str(segments_dir),
        task="liver_segments",
    )

    print("Saved raw liver segment files to:")
    print(segments_dir)


if __name__ == "__main__":
    freeze_support()
    main()