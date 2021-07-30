from fastapi import FastAPI
from core.config import settings
from api.api import api_rounter
import uvicorn

app = FastAPI(
    title=settings.PROJECT_NAME
)

app.include_router(api_rounter, prefix=settings.API_V1_STR)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=settings.PORT)
