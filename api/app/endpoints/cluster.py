import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.utils.data_processing import process_clustering
from app.models import Point

# Define a global dictionary to store the clusters
clusters_store = {}

class ClusterRequest(BaseModel):
    num_clusters: int

router = APIRouter()

@router.post("/cluster", response_model=dict)
def cluster_points(request: ClusterRequest):
    try:
        # Process clustering
        clustered_points = process_clustering(request.num_clusters)
        
        # Generate a unique ID for this cluster
        cluster_id = str(uuid.uuid4())
        
        # Store the clusters in the dictionary
        clusters_store[cluster_id] = clustered_points
        
        # Return the cluster ID
        return {"cluster_id": cluster_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clusters/{cluster_id}", response_model=List[Point])
def get_clusters(cluster_id: str):
    try:
        # Retrieve the clusters from the dictionary
        if cluster_id in clusters_store:
            return clusters_store[cluster_id]
        else:
            raise HTTPException(status_code=404, detail="Cluster ID not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))