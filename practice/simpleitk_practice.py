import SimpleITK as sitk
import numpy as np
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt

# Read the 2D image
logo = sitk.ReadImage("practice/simpleITK.jpeg")

# Access and modify a pixel
print(logo.GetPixel(0, 0))
logo.SetPixel(0,0,[255,0,0])
print(logo.GetPixel(0, 0))

# Alternative pixel access using indexing
print(logo[0, 1])
logo[0, 1] = [0, 255, 0]
print(logo[0, 1])

# Create a copy of the image
logo_copy = sitk.Image(logo)

# Get image dimensions
height = logo_copy.GetHeight()

# Flip a vertical strip of the image
logo_copy[115:190, 0:height] = logo_copy[190:115:-1, 0:height]

# Display the image
plt.imshow(sitk.GetArrayViewFromImage(logo_copy))
plt.axis("off")
plt.show()