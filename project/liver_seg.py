from multiprocessing import freeze_support
from totalsegmentator.python_api import totalsegmentator

def main():
    totalsegmentator(
        input="/Users/ne/Desktop/honours/spect-data/PRISM-WM-004/SPECT_nifti_pre/Biliary Scan/CT AC AbdoLowDose 3.0 I41s/003_AC__AbdoLowDose__3_0__I41s.nii.gz",
        output="/Users/ne/Desktop/honours/spect-data/NE ALTERED/liver_segmentation_pre004",
        task="liver_segments"
    )

if __name__ == "__main__":
    freeze_support()
    main()