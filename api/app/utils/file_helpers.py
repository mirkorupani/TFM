import json
import os
from scipy.spatial import distance

def load_hyperparams(file):
    with open(file, 'r') as f:
        return json.load(f)

def find_hyperparams(point, cluster_data, predefined_points):
    hyperparams_dir = os.path.join("static", "hyperparameters")
    hyperparam_files = [os.path.join(hyperparams_dir, f) for f in os.listdir(hyperparams_dir) if f.endswith('.json')]

    def find_hyperparams_by_distance(point, predefined_points):
        # Keep only the predefined points that have hyperparameters
        predefined_points = [p for p in predefined_points if any(f"sta{p.id}" in file for file in hyperparam_files)]
        closest_point = min(predefined_points, key=lambda p: distance.euclidean((p.latitude, p.longitude), (point.latitude, point.longitude)))
        print("Closest point", closest_point)
        file = next(file for file in hyperparam_files if f"sta{closest_point.id}" in file)
        return load_hyperparams(file)


    def find_hyperparams_by_cluster(point, cluster_data):
        # Check if there are hyperparameters for the point
        for file in hyperparam_files:
            sta = int(file.split('Hyperparameters')[0].split('sta')[-1])
            if point.id == sta:
                print("Hyperparameters found for the point")
                return load_hyperparams(file)
        
        # Check if there are hyperparameters for the cluster
        for file in hyperparam_files:
            sta = int(file.split('Hyperparameters')[0].split('sta')[-1])
            # Find cluster of the sta in cluster_data
            for point_cd in cluster_data:
                if point_cd.id == sta:
                    cluster = point_cd.cluster
                    break
            if cluster == point.cluster:
                print("Hyperparameters found for the cluster")
                return load_hyperparams(file)
        
        # If no hyperparameters found for the point or cluster, return None
        return None

    if cluster_data is not None:
        print("Cluster found", point.cluster)
        hyperparams = find_hyperparams_by_cluster(point, cluster_data)
        if not hyperparams:
            print("No hyperparameters found for the cluster, using closest point")
            hyperparams = find_hyperparams_by_distance(point, predefined_points)

    else:
        print("No cluster found, using closest point")
        hyperparams = find_hyperparams_by_distance(point, predefined_points)

    return hyperparams
