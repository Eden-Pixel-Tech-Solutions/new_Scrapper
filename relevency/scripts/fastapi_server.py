#!/usr/bin/env python3
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os

# --------------------------------------------------------
# FIX PATHS SO PYTHON CAN IMPORT global_relevancy
# --------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

sys.path.append(SCRIPTS_DIR)
sys.path.append(ROOT)

# Import global_relevancy.py
from global_relevancy import predict as global_predict

# Import analyser_relevancy if present
try:
    from analyser_relevancy import predict_relevancy as analyser_predict
except:
    analyser_predict = None

# --------------------------------------------------------
# FASTAPI APP
# --------------------------------------------------------
app = FastAPI()

# CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    query: str
    top_k: int | None = 5

# --------------------------------------------------------
# ROUTES
# --------------------------------------------------------

@app.post("/predict")
async def predict_api(data: QueryModel):
    query = data.query.strip()
    top_k = data.top_k or 5

    if not query:
        return {"error": "Query cannot be empty"}

    try:
        result = global_predict(query, top_k)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Relevancy API is running"}

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8009,
        reload=True
    )
