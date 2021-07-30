from fastapi import APIRouter
from api.endpoints import face, ocr

api_rounter = APIRouter()
api_rounter.include_router(face.rounter, prefix="/face", tags=["face"])
api_rounter.include_router(ocr.rounter, prefix="/ocr", tags=['ocr'])