from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.routes_predict import router as predict_router

app = FastAPI(
    title="AgriGuard API",
    description="AI-Based Early Crop Disease, Pest & Severity Detection System",
    version="1.0.0"
)

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1")

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "mode": os.environ.get("AGRIGUARD_MODE", "demo")}
