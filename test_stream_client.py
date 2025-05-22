import cv2
import numpy as np
import requests
import time
import logging
import os
from datetime import datetime
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_stream():
    """Test the video stream by connecting to the server and saving frames"""
    stream_url = "http://172.16.73.124:5000/video_feed"
    logger.info(f"Connecting to stream at {stream_url}")
    
    # Create output directory for frames
    output_dir = "stream_test_frames"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving frames to {output_dir}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        # Open the stream with stream=True and no timeout
        with requests.get(stream_url, stream=True, timeout=None) as response:
            logger.info(f"Stream connection established. Status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Failed to connect to stream. Status code: {response.status_code}")
                return
            
            # Process the stream
            bytes_buffer = b''
            
            # Read the response content
            for chunk in response.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                    
                bytes_buffer += chunk
                
                # Look for frame boundary
                while True:
                    # Find the start of a frame
                    a = bytes_buffer.find(b'\r\n--frame\r\n')
                    if a == -1:
                        break
                        
                    # Find the end of the headers
                    b = bytes_buffer.find(b'\r\n\r\n', a)
                    if b == -1:
                        break
                        
                    # Find the start of the next frame
                    c = bytes_buffer.find(b'\r\n--frame\r\n', b)
                    if c == -1:
                        # Need more data
                        break
                        
                    # Extract the image data
                    image_data = bytes_buffer[b+4:c]
                    bytes_buffer = bytes_buffer[c:]
                    
                    try:
                        # Convert to image
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            frame_count += 1
                            
                            # Save every 10th frame to avoid filling disk
                            if frame_count % 10 == 0:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                frame_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
                                cv2.imwrite(frame_path, frame)
                                logger.info(f"Saved frame {frame_count} to {frame_path}")
                            
                            # Log frame rate every 100 frames
                            if frame_count % 100 == 0:
                                elapsed_time = time.time() - start_time
                                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                                logger.info(f"Processed {frame_count} frames at {fps:.2f} FPS")
                            
                            # Add a small delay
                            time.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        continue
                    
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
    except Exception as e:
        logger.error(f"Error in stream: {e}", exc_info=True)
    finally:
        logger.info(f"Stream test completed. Total frames processed: {frame_count}")

if __name__ == "__main__":
    test_stream() 
