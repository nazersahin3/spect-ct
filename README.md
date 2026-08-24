# SPECT-CT Liver Segmentation Pipeline

This repository contains Python scripts for processing liver SPECT-CT data, including whole-liver segmentation, ROI segment generation (Couinaud, Dose bands), image registration, segmentation quality control, and ROI-based calculated statistics.

## Main workflow
### Image Registration 

### Segmentation 

### Segment Statistics
1. Generate whole-liver masks using TotalSegmentator.
2. Generate Couinaud liver segment masks using TotalSegmentator.
3. Combine individual segment masks into a single labelled volume.
4. Clip segment labels to the whole-liver mask.
5. Perform quality-control checks on segment coverage and labels.
6. For failed post-treatment segmentations, use pre-treatment segment labels as a patient-specific anatomical prior.
7. Transfer pre-treatment labels into post-treatment space using centroid-based liver alignment.
8. Fill remaining unlabelled post-liver voxels using nearest-neighbour label propagation.
9. Extract segment-wise statistics including voxel count, volume, mean intensity, and total function.

## Notes

Medical image files, segmentation outputs, virtual environments, and generated results are excluded from version control using `.gitignore`.




