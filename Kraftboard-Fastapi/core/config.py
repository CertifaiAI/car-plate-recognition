from typing import List
from pydantic import BaseSettings

# Setting class used to store secrets or const variables 
class Settings(BaseSettings):
    API_V1_STR: str = "/api"
    PROJECT_NAME = 'FastAPI Server for Python Scripts'
    PORT = 8000

settings = Settings()