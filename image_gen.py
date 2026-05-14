import numpy as np
import os
from scipy.ndimage import gaussian_filter
from PIL import Image

# --- CONFIG ---
NUM_IMAGES = 10
IMG_SIZE = 500
OUTPUT_DIR = "temp_images"
#OUTPUT_DIR = "blob_images"
DATASET_FILE = "temp_dataset.npy"
#DATASET_FILE = "clustering_dataset.npy"
NOISE_SMOOTHNESS = 2.0

# --- SETUP ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- GENERATOR LOGIC ---
def generate_localized_noise_blob(size, smoothness):
    cx, cy = np.random.randint(0, size, size=2)
    radius = np.random.uniform(size * 0.2, size * 0.5)
    color = np.random.uniform(0.4, 1.0, size=3) 

    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)

    mask = np.clip(1.0 - (dist / radius), 0, 1)

    #procedural smooth noise (Perlin approximation)
    raw_noise = np.random.rand(size, size)
    smooth_noise = gaussian_filter(raw_noise, sigma=smoothness)
    
    noise_min, noise_max = smooth_noise.min(), smooth_noise.max()
    smooth_noise = (smooth_noise - noise_min) / (noise_max - noise_min)

    intensity = mask * smooth_noise
    # 1. STATIC BACKGROUND
    bg = np.random.randint(0, 100, (size, size, 3)) * (1 - mask)[:, :, np.newaxis] 
    
    # 2. BLACK BACKGROUND 
    #bg = np.zeros((size, size, 3))

    img = np.zeros((size, size, 3))
    for i in range(3):
        img[:, :, i] = (bg[:, :, i] * (1 - mask)) + (intensity * color[i] * 255)

    return img.astype(np.uint8)

# --- EXECUTION ---
dataset = []

for i in range(NUM_IMAGES):
    img_data = generate_localized_noise_blob(IMG_SIZE, NOISE_SMOOTHNESS)
    
    # Save image for final video compilation
    Image.fromarray(img_data).save(os.path.join(OUTPUT_DIR, f"img_{i:05d}.png"))
    
    # Flatten (40x40x3 -> 4800) for clustering algorithms
    dataset.append(img_data.flatten())

#save flattened array for ML script
np.save(DATASET_FILE, np.array(dataset))
print(f"Generated {NUM_IMAGES} images to '{OUTPUT_DIR}'")
print(f"Dataset matrix saved to '{DATASET_FILE}'")