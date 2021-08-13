from typing import List
from numpy import array
from pydantic import BaseModel
import numpy

class faceImage(BaseModel):
    image: str

class faceImageEmbeddings(BaseModel):
    image: str
    boxes: str
    points: str

class comparisons(BaseModel):
    embeddings: List
    embedding_to_compare: str