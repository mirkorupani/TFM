import xarray as xr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

def process_clustering(n_clusters):

    # Load the configuration file
    with open("static\config.json") as f:
        config = json.load(f)

    # Load datasets
    datasets = [xr.open_dataset(file) for file in config["nc_files"]]

    # Handle overlapping datasets
    combined_ds = datasets[0]
    for ds in datasets[1:]:
        combined_ds = combined_ds.combine_first(ds)

    # Chunk the combined dataset
    combined_ds = combined_ds.chunk({"time": 1000})

    # Select the variables
    combined_ds = combined_ds[["u_x", "u_y", "waterlevel"]].sel(Layer=-0.1)

    # Drop the Layer dimension
    ds = combined_ds.drop_vars(['Layer', 'platform_name', 'x', 'y'])

    def reshape_data(var):
        df = var.to_dataframe().unstack(level='time')
        return df

    u_x_df = reshape_data(ds['u_x'])
    u_y_df = reshape_data(ds['u_y'])
    waterlevel_df = reshape_data(ds['waterlevel'])

    # Combine all dataframes into a single dataframe
    data = pd.concat([u_x_df, u_y_df, waterlevel_df], axis=1)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=config["n_components"])
    data_pca = pca.fit_transform(data_scaled)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=config["randomState"])
    kmeans.fit(data_pca)

    # Get the cluster labels
    labels = kmeans.labels_

    # Add cluster labels to predefined points
    from getPoints import getPoints
    predefined_points = getPoints()

    clustered_points = []
    for idx, point in enumerate(predefined_points):
        point_copy = point.copy()
        point_copy["cluster"] = int(labels[idx])
        clustered_points.append(point_copy)

    return clustered_points
