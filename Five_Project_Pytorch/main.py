from fastapi import FastAPI
import uvicorn
from api.model_gray_FP import router as gray_router
from api.model_rgb_FP import router as rgb_router

five_project_app = FastAPI()





five_project_app.include_router(gray_router)
five_project_app.include_router(rgb_router)















if __name__ == '__main__':
    uvicorn.run(five_project_app, host='127.0.0.1', port=8010)