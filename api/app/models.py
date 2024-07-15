from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime

class Point(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float
    cluster: Optional[int] = None

class ClusterRequest(BaseModel):
    num_clusters: int

class TrainRequest(BaseModel):
    point: Point
    cluster_data: Optional[List[Point]]

class TrainResponse(BaseModel):
    point_id: int
    x: List[datetime.datetime]
    y_u_x: List[float]
    yPred_u_x: List[float]
    y_u_y: List[float]
    yPred_u_y: List[float]
    y_waterlevel: List[float]
    yPred_waterlevel: List[float]