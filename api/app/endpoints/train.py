from fastapi import APIRouter, HTTPException
from app.models import TrainRequest, TrainResponse
from predictStation import PredictStation
from fastapi.responses import JSONResponse
from app.utils.file_helpers import load_hyperparams, find_hyperparams
from getPoints import getPoints
import json
import os
import pandas as pd
from app.models import Point

router = APIRouter()
predefined_points = getPoints()
# Convert predefined_points to a list of Point objects
predefined_points = [Point(**point) for point in predefined_points]

@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    if request.cluster_data is None:
        print("Cluster data not provided")
    else:
        print("Cluster data provided")
    
    config = json.load(open("static/config.json"))
    config["predictands"]["station"] = request.point.id

    print("Point:", request.point)
    
    hyperparams = find_hyperparams(request.point, request.cluster_data, predefined_points)
    
    predictStation = PredictStation(config)
    predictStation.train(hyperparameters=hyperparams)

    y, yPred = predictStation.predict(predMatrix=predictStation.model.predMatrix if not isinstance(predictStation.model, list) else predictStation.model[0].predMatrix)
    if isinstance(predictStation.model, list):
        for i, model in enumerate(predictStation.model):
            if i == 0:
                continue
            else:
                y = pd.concat([y, predictStation.model[i].predMatrix.yTest], axis=1)
    
    y = y.dropna()

    return TrainResponse(
        point_id=request.point.id,
        x=y.index.tolist(),
        y_u_x=y["u_x"].tolist(),
        yPred_u_x=yPred["u_x"].tolist(),
        y_u_y=y["u_y"].tolist(),
        yPred_u_y=yPred["u_y"].tolist(),
        y_waterlevel=y["waterlevel"].tolist(),
        yPred_waterlevel=yPred["waterlevel"].tolist()
    )