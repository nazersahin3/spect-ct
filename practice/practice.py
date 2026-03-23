import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load a sample image
image = data.camera()

print("Image shape:", image.shape)
print("Data type:", image.dtype)

plt.imshow(image, cmap="gray")
plt.title("Sample Image")
plt.axis("off")
plt.show()

