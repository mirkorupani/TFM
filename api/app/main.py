import sys

# Add the root directory to the Python path
sys.path.append(r"C:\TFM")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.endpoints import cluster, points, train
from fastapi.middleware.cors import CORSMiddleware

# Add this line before defining your FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cluster.router)
app.include_router(points.router)
app.include_router(train.router)

# Mount static files directory at root endpoint
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)