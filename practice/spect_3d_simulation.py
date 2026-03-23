import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Create 3D grid
size = 64
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
z = np.linspace(-1, 1, size)

X, Y, Z = np.meshgrid(x, y, z)

# Create synthetic "tumour" activity (3D Gaussian blob)
activity = np.exp(-(X**2 + Y**2 + Z**2) * 10)

print("Volume shape:", activity.shape)

# Simulate system resolution (blur)
blurred = gaussian_filter(activity, sigma=2)

# Simulate projections (very simplified)
projections = np.sum(blurred, axis=2)

# Scale to counts
scaled_proj = projections / projections.max() * 100
noisy_proj = np.random.poisson(scaled_proj)

# Naive backprojection
reconstruction = np.repeat(noisy_proj[:, :, np.newaxis], size, axis=2)

plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.imshow(activity[:, :, size//2], cmap="gray")
plt.title("True Activity")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(blurred[:, :, size//2], cmap="gray")
plt.title("Blurred")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(projections, cmap="gray")
plt.title("Projection")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(noisy_proj, cmap="gray")
plt.title("Noisy Projection")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(reconstruction[:, :, size//2], cmap="gray")
plt.title("Backprojection")
plt.axis("off")

plt.tight_layout()
plt.show()