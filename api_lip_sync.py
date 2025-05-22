from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging
import subprocess
import uuid
import torch
from gtts import gTTS
import tempfile
import threading
from typing import Optional, Dict, Set
import asyncio
import aiofiles
import json
from datetime import datetime
import base64
import cv2
import numpy as np
from queue import Queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lip Sync API",
    description="API for text-to-speech and lip sync generation with WebSocket support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        # Processing status for each connection
        self.processing_status: Dict[str, dict] = {}
        # Video frame queues for each client
        self.frame_queues: Dict[str, Queue] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.processing_status[client_id] = {
            "status": "connected",
            "message": "Connected to server",
            "timestamp": datetime.now().isoformat()
        }
        self.frame_queues[client_id] = Queue(maxsize=30)  # Buffer 30 frames
        await self.send_status(client_id)
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.processing_status:
            del self.processing_status[client_id]
        if client_id in self.frame_queues:
            del self.frame_queues[client_id]
            
    async def send_status(self, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(
                self.processing_status[client_id]
            )
            
    async def update_status(self, client_id: str, status: str, message: str):
        if client_id in self.processing_status:
            self.processing_status[client_id].update({
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            await self.send_status(client_id)
            
    def add_frame(self, client_id: str, frame):
        """Add a frame to the client's queue"""
        if client_id in self.frame_queues:
            try:
                self.frame_queues[client_id].put_nowait(frame)
            except Queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queues[client_id].get_nowait()
                    self.frame_queues[client_id].put_nowait(frame)
                except:
                    pass

    async def send_frame(self, client_id: str, frame):
        """Send a frame to the client via WebSocket"""
        if client_id in self.active_connections:
            try:
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame as JSON
                await self.active_connections[client_id].send_json({
                    "type": "frame",
                    "data": frame_bytes,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending frame to client {client_id}: {e}")

# Initialize connection manager
manager = ConnectionManager()

class LipSyncAPI:
    def __init__(self, 
                 model_path="checkpoints",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        
        # Create temp directory for processing
        self.temp_dir = "temp_processing"
        os.makedirs(self.temp_dir, exist_ok=True)

    def text_to_speech(self, text, lang='en'):
        """Convert text to speech using gTTS"""
        try:
            logger.info(f"Converting text to speech: {text}")
            
            # Generate unique filename
            audio_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}.wav")
            
            # Convert text to speech
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(audio_path)
            
            logger.info(f"Text-to-speech conversion completed: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            raise

    async def run_inference(self, client_id: str, audio_path: str, image_path: str, batch_size: int = 8):
        """Run the inference command with status updates and frame streaming"""
        try:
            cmd = [
                "python", "inference.py",
                "--driven_audio", audio_path,
                "--source_image", image_path,
                "--enhancer", "gfpgan",
                "--batch_size", str(batch_size)
            ]
            
            logger.info(f"Running inference command: {' '.join(cmd)}")
            await manager.update_status(client_id, "processing", "Starting inference...")
            
            # Create a pipe to capture the output video path
            output_dir = os.path.join("results", datetime.now().strftime("%Y_%m_%d_%H.%M.%S"))
            os.makedirs(output_dir, exist_ok=True)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Process output and stream frames
            video_path = None
            while True:
                # Read inference output
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(output.strip())
                    await manager.update_status(client_id, "processing", output.strip())
                    
                    # Look for the generated video path in the output
                    if "The generated video is named" in output:
                        video_path = output.strip().split("named ")[-1]
                        logger.info(f"Found generated video at: {video_path}")
                        
                        # Start streaming the generated video
                        if os.path.exists(video_path):
                            cap = cv2.VideoCapture(video_path)
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                    
                                # Add frame to queue and send via WebSocket
                                manager.add_frame(client_id, frame)
                                await manager.send_frame(client_id, frame)
                                
                                # Small delay to control frame rate
                                await asyncio.sleep(0.033)  # ~30 FPS
                            
                            cap.release()
            
            # Check for errors
            if process.returncode != 0:
                error = process.stderr.read()
                raise Exception(f"Inference failed: {error}")
                
            await manager.update_status(client_id, "completed", "Inference completed successfully")
                
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            await manager.update_status(client_id, "error", str(e))
            raise

    def cleanup(self, audio_path, image_path):
        """Clean up temporary files"""
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")

    async def process_request(self, client_id: str, text: str, image_path: str, batch_size: int = 8, lang: str = 'en'):
        """Process a complete request: TTS -> Lip Sync -> Stream"""
        audio_path = None
        try:
            # Step 1: Convert text to speech
            await manager.update_status(client_id, "processing", "Converting text to speech...")
            audio_path = self.text_to_speech(text, lang)
            
            # Step 2: Run inference and stream frames
            await self.run_inference(client_id, audio_path, image_path, batch_size)
            
            return {
                "status": "success",
                "message": "Processing completed"
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            await manager.update_status(client_id, "error", str(e))
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            # Clean up in a separate thread to not block the response
            if audio_path:
                threading.Thread(
                    target=self.cleanup,
                    args=(audio_path, None)
                ).start()

# Initialize the API
lip_sync_api = LipSyncAPI()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "generate":
                    # Handle generation request
                    text = message.get("text")
                    image_data = message.get("image")  # Base64 encoded image
                    batch_size = message.get("batch_size", 8)
                    lang = message.get("lang", "en")
                    
                    # Save image from base64
                    image_path = os.path.join(lip_sync_api.temp_dir, f"{uuid.uuid4()}.png")
                    image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                    async with aiofiles.open(image_path, 'wb') as f:
                        await f.write(image_bytes)
                    
                    # Process request
                    await lip_sync_api.process_request(
                        client_id=client_id,
                        text=text,
                        image_path=image_path,
                        batch_size=batch_size,
                        lang=lang
                    )
                    
            except json.JSONDecodeError:
                await manager.update_status(client_id, "error", "Invalid JSON message")
            except Exception as e:
                await manager.update_status(client_id, "error", str(e))
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/generate")
async def generate(
    client_id: str = Form(...),
    text: str = Form(...),
    image: UploadFile = File(...),
    batch_size: int = Form(8),
    lang: str = Form('en')
):
    """Generate lip-synced video from text and image"""
    try:
        # Save image file
        image_path = os.path.join(lip_sync_api.temp_dir, f"{uuid.uuid4()}.png")
        async with aiofiles.open(image_path, 'wb') as f:
            content = await image.read()
            await f.write(content)
            
        # Process request
        result = await lip_sync_api.process_request(
            client_id=client_id,
            text=text,
            image_path=image_path,
            batch_size=batch_size,
            lang=lang
        )
        
        if result["status"] == "success":
            return JSONResponse(content=result, status_code=200)
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{client_id}")
async def status(client_id: str):
    """Check processing status for a specific client"""
    if client_id in manager.processing_status:
        return manager.processing_status[client_id]
    return {"status": "not_found", "message": "Client not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_lip_sync:app", host="0.0.0.0", port=8002, reload=True) 
