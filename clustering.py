import pandas as pd
from k_means_constrained import KMeansConstrained
import numpy as np
import json
import math
import poi_get  # Assuming this is your file containing the API logic

# --- 1. Define Input and Configuration ---

# Get data from external source
# (Ensure your poi_get.py is in the same directory and configured correctly)
input_data = poi_get.run_travel_planner_external(
    "Udaipur",
    3,
    ["City Palace, Udaipur", "Jagdish Temple"],
    ["history", "lakes"],
    ["sports", "entertainment"]
)

# Configuration
K = 3  # Number of Days (Clusters)

print(f"--- Processing Data for {K} Days ---")

# --- 2. Data Preparation ---

# Convert places to DataFrame
df = pd.DataFrame(input_data['places'])
total_places = len(df)

# Ensure required columns exist
required_cols = ['score', 'is_mandatory', 'latitude', 'longitude']
for col in required_cols:
    if col not in df.columns:
        if col == 'is_mandatory': df[col] = False
        elif col == 'latitude': df[col] = 0.0
        elif col == 'longitude': df[col] = 0.0

# Extract Features for Clustering
X = df[['latitude', 'longitude']]

# --- 3. Dynamic Constraint Calculation ---

avg_places_per_day = total_places // K
size_min = max(1, int(avg_places_per_day * 0.4)) 
size_max = math.ceil(avg_places_per_day * 0.6)

# Safety buffer for constraints
if K * size_min > total_places:
    size_min = total_places // K
if K * size_max < total_places:
    size_max = math.ceil(total_places / K) + 1

# --- 4. Clustering ---

clf = KMeansConstrained(
    n_clusters=K,
    size_min=size_min,
    size_max=size_max,
    random_state=42
)

df['cluster_label'] = clf.fit_predict(X)
cluster_centers = clf.cluster_centers_

# --- 5. Nearest Neighbors Calculation (New Logic) ---

def get_nearest_clusters(current_cluster_id, all_centers, top_n=2):
    """
    Calculates distances from the current cluster to all others 
    and returns the IDs of the nearest 'top_n' clusters.
    """
    distances = []
    current_center = all_centers[current_cluster_id]
    
    for other_id, other_center in enumerate(all_centers):
        if current_cluster_id == other_id:
            continue
        
        # Calculate Euclidean distance
        dist = np.linalg.norm(current_center - other_center)
        distances.append((other_id, dist))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    
    # Return just the IDs of the top N closest clusters
    return [d[0] for d in distances[:top_n]]

# --- 6. Constructing the Final Structured Output ---

# A. Places Data
places_output = df.to_dict(orient='records')

# B. Centroid Data
avg_center_lat = np.mean(cluster_centers[:, 0])
avg_center_long = np.mean(cluster_centers[:, 1])
centroid_output = {
    "avg_lat": float(avg_center_lat),
    "avg_long": float(avg_center_long)
}

# C. Clustering Data
clustering_output = {}

grouped = df.groupby('cluster_label')

for cluster_id, group_df in grouped:
    # 1. Basic Lists
    place_ids = group_df['id'].tolist()
    
    # 2. Metrics
    avg_score = round(group_df['smart_score'].mean(), 2)
    mandatory_count = int(group_df['is_mandatory'].sum())
    
    # 3. Nearest Clusters (New Function Call)
    # Convert cluster_id to int for indexing numpy array
    
    clustering_output[str(cluster_id)] = {
        "place_ids": place_ids,
        "stats": {
            "avg_score": avg_score,
            "mandatory_count": mandatory_count
        }
    }

# --- 7. Final Assembly and Print ---

final_structure = {
    "Places": places_output,
    "centroid": centroid_output,
    "clustering": clustering_output
}

print("\n" + "="*50)
print("## ✅ Final Formatted JSON Output")
print("="*50)

# default=float handles numpy data types automatically
print(json.dumps(final_structure, indent=4, default=float))