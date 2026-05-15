import uvicorn
from recommendation_service.api.main import app

if __name__ == "__main__":
    uvicorn.run("recommendation_service.api.main:app", port=8000, reload=True)
