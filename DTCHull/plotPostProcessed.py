import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

archs = ["AE", "AAE", "VAE", "BEGAN"]
facecolor=["r", "b", "g", "c"]
cut = 3000

datf_data = np.load("postProcessingForce_"+"data"+".npy")
print(datf_data.shape)
total_data = np.mean(np.linalg.norm(datf_data[:, cut:, 1:4], axis=2, keepdims=True), axis=1)
drag_data = np.mean(datf_data[:, cut:, 1], axis=1)

datf_data = np.load("postProcessingMomentum_"+"data"+".npy")
momx_data = np.mean(datf_data[:, cut:, 1], axis=1)

    
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, arch in enumerate(archs):
    datf = np.load("postProcessingForce_"+arch+".npy")
    
    total = np.mean(np.linalg.norm(datf[:, cut:, 1:4], axis=2, keepdims=True), axis=1)
    drag = np.mean(datf[:, cut:, 1], axis=1)
    
    datf = np.load("postProcessingMomentum_"+arch+".npy")
    momx = np.mean(datf[:, cut:, 1], axis=1)

    axes[i//2, i%2].hist(momx, 50, density=True, facecolor=facecolor[i], alpha=0.75, label=arch)
    axes[i//2, i%2].hist(momx_data, 50, density=True, facecolor="y", alpha=0.75, label="VP-FFD")
    
    axes[i//2, i%2].grid(which="both")
    axes[i//2, i%2].set_title(arch)
    axes[i//2, i%2].legend()

plt.suptitle("momentum x component")
plt.show()