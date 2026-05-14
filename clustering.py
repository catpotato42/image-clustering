import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

# --- CONFIG ---
DATASET_FILE = "clustering_dataset.npy"
N_CLUSTERS = 30

# --- LOAD ---
data = np.load(DATASET_FILE).astype('float32') / 255.0
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(f"Data loaded. Shape: {data_scaled.shape}")

# --- K-MEANS ---
print("Running K-Means...")
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# --- DBSCAN ---
print("Running DBSCAN...")
dbscan = DBSCAN(eps=40, min_samples=5) 
dbscan_labels = dbscan.fit_predict(data_scaled)

# --- AUTOENCODER ---
print("Training Neural Network...")
input_dim = data_scaled.shape[1] # 2700
encoding_dim = 16 

input_layer = tf.keras.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
bottleneck = tf.keras.layers.Dense(encoding_dim, activation='relu', name="bottleneck")(encoded)
decoded = tf.keras.layers.Dense(128, activation='relu')(bottleneck)
output_layer = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = tf.keras.Model(input_layer, output_layer)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data_scaled, data_scaled, epochs=15, batch_size=256, verbose=0)

encoder = tf.keras.Model(inputs=input_layer, outputs=bottleneck)

print("Extracting neural features...")
encoded_features = encoder.predict(data_scaled)
nn_labels = KMeans(n_clusters=N_CLUSTERS, n_init=10).fit_predict(encoded_features)

# --- HIERARCHICAL CLUSTERING ---
print("Running Hierarchical Clustering...")

hc = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=150, 
    linkage='ward'
)
hc_labels = hc.fit_predict(data_scaled)
n_found = len(np.unique(hc_labels))
print(f"Hierarchical complete. Found {n_found} clusters.")

# --- EVALUATION ---
def get_score(name, labels):
    mask = labels != -1 #ignore DBSCAN noise
    if len(np.unique(labels[mask])) > 1:
        score = silhouette_score(data_scaled[mask], labels[mask])
        print(f"{name} Silhouette Score: {score:.4f}")
    else:
        print(f"{name}: Failed to find distinct clusters.")

get_score("K-Means", kmeans_labels)
get_score("DBSCAN", dbscan_labels)
get_score("Neural Net", nn_labels)
get_score("Hierarchical", hc_labels)

# --- SAVE ---
np.savez("cluster_results.npz", 
         kmeans=kmeans_labels, 
         dbscan=dbscan_labels, 
         nn=nn_labels, 
         hc=hc_labels)
print("Saved labels to cluster_results.npz")