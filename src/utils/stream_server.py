from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import threading
import queue
import time
import logging
import sys
from typing import Generator
from datetime import datetime

# Set up logging to also print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stream_server.log')
    ]
)
logger = logging.getLogger(__name__)

class StreamServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.frame_queue = queue.Queue(maxsize=30)  # Limit queue size to prevent memory issues
        self.streaming_active = False
        self.active_clients = 0
        self.frame_count = 0
        self.stream_lock = threading.Lock()
        self.last_frame_time = time.time()
        self.fps = 0
        
        # Register routes
        self.app.route('/video_feed')(self.video_feed)
        self.app.route('/status')(self.status)
        
        # Start server in a separate thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        logger.info(f"Starting streaming server on {host}:{port}")
        logger.info(f"Streaming server started at http://{host}:{port}/video_feed")

    def _run_server(self):
        """Run the Flask server"""
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def video_feed(self):
        """Video streaming route"""
        client_ip = request.remote_addr
        logger.info(f"New client connected from {client_ip}")
        
        def generate_frames():
            try:
                with self.stream_lock:
                    self.active_clients += 1
                    self.streaming_active = True
                    logger.info(f"Streaming activated for client {client_ip}. Active clients: {self.active_clients}")
                
                while True:
                    try:
                        # Get frame from queue with timeout
                        frame = self.frame_queue.get(timeout=1.0)
                        
                        # Update FPS calculation
                        current_time = time.time()
                        time_diff = current_time - self.last_frame_time
                        if time_diff > 0:
                            self.fps = 1.0 / time_diff
                        self.last_frame_time = current_time
                        
                        # Encode frame
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if not ret:
                            logger.error("Failed to encode frame")
                            continue
                            
                        frame_bytes = buffer.tobytes()
                        self.frame_count += 1
                        
                        # Log frame sending
                        logger.debug(f"Sending frame {self.frame_count} to client {client_ip}")
                        
                        # Yield frame with proper MIME type and headers
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                               b'\r\n' + frame_bytes + b'\r\n')
                               
                    except queue.Empty:
                        # No frames available, send a blank frame to keep connection alive
                        blank_frame = np.zeros((256, 256, 3), dtype=np.uint8)
                        ret, buffer = cv2.imencode('.jpg', blank_frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                                   b'\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.1)  # Small delay to prevent CPU spinning
                        
            except GeneratorExit:
                logger.info(f"Client {client_ip} disconnected")
            except Exception as e:
                logger.error(f"Error in generate_frames for client {client_ip}: {e}", exc_info=True)
            finally:
                with self.stream_lock:
                    self.active_clients -= 1
                    if self.active_clients == 0:
                        self.streaming_active = False
                        logger.info(f"Streaming deactivated. No active clients.")
                    else:
                        logger.info(f"Client {client_ip} disconnected. Active clients: {self.active_clients}")

        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )

    def status(self):
        """Get server status"""
        with self.stream_lock:
            status = {
                'streaming_active': self.streaming_active,
                'active_clients': self.active_clients,
                'frames_queued': self.frame_queue.qsize(),
                'total_frames_streamed': self.frame_count,
                'fps': round(self.fps, 2) if self.fps > 0 else 0
            }
        return jsonify(status)

    def add_frame(self, frame):
        """Add a frame to the queue"""
        try:
            if not self.streaming_active:
                logger.warning("Trying to add frame but streaming is not active")
                return False
                
            if self.frame_queue.full():
                # If queue is full, remove oldest frame
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            self.frame_queue.put(frame, block=False)
            return True
            
        except Exception as e:
            logger.error(f"Error adding frame to queue: {e}")
            return False

    def stop(self):
        """Stop the streaming server"""
        with self.stream_lock:
            self.streaming_active = False
            # Clear the frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
        logger.info("Stopping streaming")
        # Note: Flask server will continue running until the process exits

# Create global instance
stream_server = StreamServer()

def get_stream_server():
    """Get the global stream server instance"""
    return stream_server

def start_server(host='0.0.0.0', port=5000):
    """Start the streaming server"""
    global stream_server
    stream_server = StreamServer(host=host, port=port)
    return stream_server

def add_frame(frame):
    """Add a frame to the streaming queue"""
    return stream_server.add_frame(frame)

def stop_streaming():
    """Stop the streaming process"""
    stream_server.stop() 
