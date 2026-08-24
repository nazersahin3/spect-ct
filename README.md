# SPECT-CT Liver Pipeline

This repository contains Python scripts for processing liver SPECT-CT and radiation therapy imaging data for regional liver function analysis. The workflow includes image registration, whole-liver and Couinaud segmentation, radiation dose-band generation, segmentation quality control, SPECT scaling, and ROI-based statistical evaluation.

## Main workflow
### Image Registration 
PRE- and POST-treatment CT and SPECT images are aligned to the radiation therapy planning CT (RTPLAN CT) to provide a common spatial reference for regional analysis.

The registration workflow includes:

  Rigid registration of PRE- and POST-treatment CT images to the RT planning CT.
  Application of the corresponding rigid transformations to associated imaging data.
  Structure-guided deformable registration using the liver contour to improve local liver alignment.
  Application of the registration transformations to the corresponding SPECT images.
  Visual and quantitative quality control of the registered images and liver masks.

The resulting datasets are represented in the RT planning coordinate system so that anatomical, functional, and radiation dose information can be compared spatially.

### Segmentation 
#### Liver Couinaud Segmentation
Couinaud liver segments are generated from the RT planning CT using TotalSegmentator.

The segmentation workflow includes:

Generate a whole-liver mask using TotalSegmentator.
Generate Couinaud liver segment masks using the liver_segments TotalSegmentator task.
Combine individual Couinaud segment masks into a labelled liver volume where required.
Restrict or clip segment labels to the whole-liver mask.
Perform visual and quantitative segmentation quality control.

The resulting masks represent Couinaud liver segments I–VIII and are used for anatomical region-based functional analysis.

#### Dose Band Segmentation
Radiation therapy dose data are extracted from DICOM RTDOSE files and converted into image-based dose maps.

Dose regions are generated in 5 Gy intervals:

<5 Gy
5–10 Gy
10–15 Gy
15–20 Gy
20–25 Gy
25–30 Gy
≥30 Gy

Two forms of dose segmentation are generated:

Non-overlapping dose bands, used for regional dose-response analysis.
Cumulative dose masks (≥5, ≥10, ≥15, ≥20, ≥25 and ≥30 Gy), used for functional dose-burden analysis.

Dose masks can subsequently be restricted to the liver volume for liver-specific analysis.

### SPECT Processing
Registered SPECT images are scaled using patient-specific total liver function (TLF, %/min) measurements.

The spatial distribution of SPECT counts is used to determine the fraction of whole-liver function contained within each region of interest. This allows regional functional measurements to be expressed relative to the patient's measured total liver function.

For a given ROI:

\frac{\text{ROI sum counts}}
{\text{whole-liver sum counts}}
]

and regional liver function is calculated as:

\text{ROI fraction of liver counts}
\times
\text{patient TLF}.
]

### Segment Statistics and Evaluation

ROI-based statistics are extracted from the registered and scaled SPECT images using the Couinaud and radiation dose masks.

Calculated statistics include:

ROI voxel count
ROI volume (cm³)
ROI sum counts
Whole-liver sum counts
ROI fraction of whole-liver counts
Patient TLF (%/min)
Regional ROI TLF (%/min)
Minimum SPECT value
Maximum SPECT value
Mean SPECT value
Standard deviation
5th percentile
95th percentile
Median

Results are stored in Pandas dataframes for subsequent statistical analysis and visualisation.

The evaluation pipeline supports investigation of:

PRE- to POST-treatment regional functional changes
Dose-dependent changes in regional liver function
Functional changes within individual Couinaud segments
Functional dose burden
Relationships between regional radiation dose and subsequent functional response

### Analysis Regions 
Two complementary regional approaches are used:

Couinaud regions provide an anatomical assessment of regional liver function.
Radiation dose bands provide a dose-defined assessment of functional response, allowing changes in SPECT-derived function to be evaluated across increasing radiation dose levels.
Cumulative dose regions can additionally be used to determine the proportion of baseline functioning liver exposed to specified radiation dose thresholds.

## Notes

Medical imaging files and generated outputs are not stored in this repository.

The following are excluded from version control using .gitignore:

DICOM files
NIfTI images
Patient-specific imaging data
Segmentation outputs
Generated analysis results
Virtual environments
Temporary Python files

This repository therefore contains the processing and analysis code only and does not contain patient imaging data.




