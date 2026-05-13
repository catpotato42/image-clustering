import cv2
import numpy as np
import os

# --- CONFIG ---
RESULTS_FILE = "cluster_results.npz"
IMG_DIR = "blob_images"
FPS = 20
ALGO_TO_PLAY = "kmeans"  # Options: 'kmeans', 'dbscan', or 'nn'
DISPLAY_SIZE = (500, 500) #upscale to 500x500 for visibility

# --- LOAD DATA ---
if not os.path.exists(RESULTS_FILE):
    print(f"Error: {RESULTS_FILE} not found. Run your clustering script first!")
    exit()

data = np.load(RESULTS_FILE)
labels = data[ALGO_TO_PLAY]
unique_clusters = sorted(np.unique(labels))

# --- PLAYBACK LOOP ---
cv2.namedWindow("Clustered Playback")

for cluster_id in unique_clusters:
    #skip DBSCAN noise (-1) or play it as its own group
    cluster_name = "Noise" if cluster_id == -1 else f"Cluster {int(cluster_id) + 1}"
    
    indices = np.where(labels == cluster_id)[0]
    print(f"Playing {cluster_name} ({len(indices)} frames)...")

    for idx in indices:
        img_path = os.path.join(IMG_DIR, f"img_{idx:05d}.png")
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue

        #upscale for visibility
        frame_large = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(cluster_name, font, text_scale, thickness)[0]
        
        # Calculate X to align to the right side
        text_x = DISPLAY_SIZE[0] - text_size[0] - 20
        text_y = 40
        
        #shadow for readability, then white text
        cv2.putText(frame_large, cluster_name, (text_x+2, text_y+2), font, text_scale, (0,0,0), thickness, cv2.LINE_AA)
        cv2.putText(frame_large, cluster_name, (text_x, text_y), font, text_scale, (255,255,255), thickness, cv2.LINE_AA)

        #show frame
        cv2.imshow("Clustered Playback", frame_large)

        #press 'q' to skip to next cluster or exit.
        if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
            break

print("Playback finished.")
cv2.destroyAllWindows()