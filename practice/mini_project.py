import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import gaussian

# Step 1: Load base image
image = data.camera()

# Step 2: Simulate SPECT blur
blurred = gaussian(image, sigma=5)

# Step 3: Simulate Poisson noise
scaled = blurred / blurred.max() * 50
noisy = np.random.poisson(scaled)

# Step 4: Display
plt.figure()
plt.imshow(noisy, cmap="gray")
plt.title("Simulated SPECT-like Image")
plt.axis("off")
plt.show()