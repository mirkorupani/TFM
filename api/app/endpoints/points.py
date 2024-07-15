from fastapi import APIRouter
from getPoints import getPoints
from typing import List

router = APIRouter()

@router.get("/points", response_model=List[dict])
def get_points():
    return getPoints()