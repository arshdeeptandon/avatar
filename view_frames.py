from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import socket
import logging
import webbrowser
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameViewer(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="stream_test_frames", **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format%args}")

def get_server_ip():
    """Get the server's IP address"""
    try:
        # Create a socket to get the IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"Error getting server IP: {e}")
        return "0.0.0.0"

def run_server(port=8000):
    server_ip = get_server_ip()
    server_address = ('0.0.0.0', port)  # Listen on all interfaces
    httpd = HTTPServer(server_address, FrameViewer)
    
    # Get the server's public IP (if available)
    try:
        import requests
        public_ip = requests.get('https://api.ipify.org').text
    except:
        public_ip = server_ip
    
    # Print connection information
    logger.info("=" * 50)
    logger.info("Frame Viewer Server Started")
    logger.info("=" * 50)
    logger.info(f"Local URL: http://localhost:{port}")
    logger.info(f"Server URL: http://{server_ip}:{port}")
    if public_ip != server_ip:
        logger.info(f"Public URL: http://{public_ip}:{port}")
    logger.info("=" * 50)
    logger.info("IMPORTANT: Make sure port 8000 is open in your EC2 security group!")
    logger.info("=" * 50)
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    if not os.path.exists("stream_test_frames"):
        logger.error("stream_test_frames directory not found!")
        exit(1)
        
    # List available frames
    frames = sorted([f for f in os.listdir("stream_test_frames") if f.endswith('.jpg')])
    logger.info(f"Found {len(frames)} frames")
    if frames:
        logger.info("First few frames:")
        for f in frames[:5]:
            logger.info(f"  {f}")
    
    # Try to install requests if not available
    try:
        import requests
    except ImportError:
        logger.info("Installing requests package...")
        import subprocess
        subprocess.check_call(["pip", "install", "requests"])
        import requests
    
    run_server() 
