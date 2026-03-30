import SimpleITK as sitk

# ----------------------------
# Load CT
# ----------------------------
ct_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/data/002/CT - pre #2/003_CT_Liver_3mm_I41s_pre#2.nii"
ct = sitk.ReadImage(ct_path, sitk.sitkFloat32)

print("Size:", ct.GetSize())

# ----------------------------
# Soft tissue threshold
# ----------------------------
mask = sitk.BinaryThreshold(
    ct,
    lowerThreshold = 40,
    upperThreshold = 90,
    insideValue=1,
    outsideValue=0
)

# ----------------------------
# Crop to rough liver region
# ----------------------------
size = ct.GetSize()   # (x, y, z)

x_start = 0
x_end   = int(size[0] * 0.58)

y_start = int(size[1] * 0.10)
y_end   = int(size[1] * 0.68)

z_start = int(size[2] * 0.10)
z_end   = int(size[2] * 0.60)

roi = sitk.Image(size, sitk.sitkUInt8)
roi.CopyInformation(ct)

roi_array = sitk.GetArrayFromImage(roi)   # z, y, x
roi_array[z_start:z_end, y_start:y_end, x_start:x_end] = 1
roi = sitk.GetImageFromArray(roi_array)
roi.CopyInformation(ct)

mask = sitk.And(mask, roi)

# ----------------------------
# Clean up
# ----------------------------
mask = sitk.BinaryMorphologicalClosing(mask, [3,3,3])
mask = sitk.BinaryFillhole(mask)

# ----------------------------
# Largest connected component
# ----------------------------
cc = sitk.ConnectedComponent(mask)
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

labels = stats.GetLabels()
if not labels:
    raise RuntimeError("No components found in ROI. Try widening threshold or crop.")

largest_label = max(labels, key=lambda l: stats.GetPhysicalSize(l))
liver_mask = sitk.BinaryThreshold(cc, largest_label, largest_label)

liver_mask = sitk.BinaryMorphologicalClosing(liver_mask, [5,5,5])
liver_mask = sitk.BinaryFillhole(liver_mask)

# ----------------------------
# Save
# ----------------------------
output_path = "/Users/nana/Desktop/HONOURS/spect-ct-imac/project/results/liver_mask_pre_FINAL.nii.gz"
sitk.WriteImage(liver_mask, output_path)

print("ROI x:", x_start, x_end)
print("ROI y:", y_start, y_end)
print("ROI z:", z_start, z_end) 

print("Saved liver mask to:", output_path)