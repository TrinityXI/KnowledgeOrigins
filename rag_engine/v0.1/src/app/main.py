from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any
from RAG import *


app = FastAPI(title="Query API", version="1.0.0")
class QueryInput(BaseModel):
    data: Dict[str, Any]
class QueryOutput(BaseModel):
    result: Dict[str, Any]
    status: str = "success"
@app.post("/query", response_model=QueryOutput)
async def query(input_data: QueryInput):
    rt = search(input_data.data)
    processed_data = {
        "received_data": rt,
        "processed": True,
        "message": "Data processed successfully"
    }

    return QueryOutput(result=processed_data)
@app.get("/")
async def root():
    return {"message": "Query API is running", "status": "ok"}

uvicorn.run(app, host="0.0.0.0", port=8100)


