import sys
import os
# Get the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from getPoints import getPoints
import json
import os
import sys
from predictStation import PredictStation
import pandas as pd
import datetime


app = FastAPI()

# Dummy data: Replace with your actual data points
predefined_points = getPoints()

# Request model
class TrainRequest(BaseModel):
    point_id: int

# Response model
class TrainResponse(BaseModel):
    point_id: int
    x: List[datetime.datetime]
    y_u_x: List[float]
    yPred_u_x: List[float]
    y_u_y: List[float]
    yPred_u_y: List[float]
    y_waterlevel: List[float]
    yPred_waterlevel: List[float]

@app.get("/points", response_model=List[dict])
def get_points():
    return predefined_points

@app.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    point = next((p for p in predefined_points if p["id"] == request.point_id), None)
    if not point:
        return JSONResponse(status_code=404, content={"message": "Point not found"})
    
    config = json.load(open("config.json"))
    config["predictands"]["station"] = point["id"] - 1
    
    predictStation = PredictStation(config)

    # Train model
    hyperparameters = json.load(open("hyperparameters.json"))
    predictStation.train(hyperparameters=hyperparameters)

    # Predict test data
    y, yPred = predictStation.predict(predMatrix=predictStation.model.predMatrix if not isinstance(predictStation.model, list) else predictStation.model[0].predMatrix)

    if isinstance(predictStation.model, list):
        # Merge y variables
        for i, model in enumerate(predictStation.model):
            if i == 0:
                continue
            else:
                y = pd.concat([y, predictStation.model[i].predMatrix.yTest], axis=1)
    
    # Remove rows with NaN values
    y = y.dropna()

    return TrainResponse(
    point_id=request.point_id,
    x=y.index.tolist(), 
    y_u_x=y["u_x"].tolist(),
    yPred_u_x=yPred["u_x"].tolist(),
    y_u_y=y["u_y"].tolist(),
    yPred_u_y=yPred["u_y"].tolist(),
    y_waterlevel=y["waterlevel"].tolist(),
    yPred_waterlevel=yPred["waterlevel"].tolist()
)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

