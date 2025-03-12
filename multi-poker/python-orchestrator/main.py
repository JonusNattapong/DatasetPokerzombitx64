import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import requests
import json
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Configuration ---
GO_SERVICE_URL = os.environ.get("GO_SERVICE_URL", "http://localhost:8080")

# --- Helper Functions ---
async def call_go_service(endpoint: str, data=None, method="POST"):
    url = f"{GO_SERVICE_URL}{endpoint}"
    logger.info(f"Calling Go service at {url}")
    if method == "POST":
        response = await asyncio.to_thread(requests.post, url, json=data)
    elif method == "GET":
        response = await asyncio.to_thread(requests.get, url)
    else:
        raise ValueError("Invalid method")

    if response.status_code != 200 and response.status_code != 202:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response

# --- API Endpoints ---
@app.post("/process_hand_history/")
async def process_hand_history(file: UploadFile = File(...)):
    """
    Upload a hand history file for processing.
    Sends the file to the Go service for asynchronous processing.
    """
    try:
        logger.info("Processing hand history file")
        contents = await file.read()
        hand_history_data = json.loads(contents)  # Assuming JSON format
        
        # Send to Go service
        response = await call_go_service("/process", data=hand_history_data)
        return {"message": "Hand history processing initiated", "hand_id": hand_history_data.get("id")}

    except Exception as e:
        logger.error(f"Error processing hand history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hand/{hand_id}")
async def get_hand_history(hand_id: str):
    """
    Retrieve a processed hand history by its ID.
    """
    try:
        logger.info(f"Retrieving hand history for ID: {hand_id}")
        response = await call_go_service(f"/hand/{hand_id}", method="GET")
        return response.json()
    except HTTPException as e:
        if e.status_code == 404:
            logger.warning(f"Hand history not found for ID: {hand_id}")
            raise HTTPException(status_code=404, detail="Hand history not found")
        else:
            logger.error(f"Error retrieving hand history: {str(e)}")
            raise

@app.post("/go_evaluate")
async def go_evaluate(cards: list[str]):
    """
    Evaluate a poker hand using the Go service.
    """
    try:
        logger.info(f"Sending hand to Go service for evaluation: {cards}")
        
        # Convert card strings to Card objects
        card_objects = [{"rank": card[:1], "suit": card[1:]} for card in cards]

        response = await call_go_service("/evaluate", data=card_objects, method="POST")
        return response.json()
    except Exception as e:
        logger.error(f"Error evaluating hand with Go service: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Python orchestrator")

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
