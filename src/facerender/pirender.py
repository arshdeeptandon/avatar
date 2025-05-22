import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.utils.face_enhancer import enhancer_generator_with_len
from src.utils.videoio import save_video_with_watermark

class PIRender(nn.Module):
    def __init__(self, init_path, device):
        super(PIRender, self).__init__()
        self.device = device
        self.checkpoint_path = os.path.join(init_path, 'checkpoints', 'epoch_00190_iteration_000400000_checkpoint.pt')
        
        # Load PIRender model
        self.model = self.load_model()
        self.model.eval()
        self.model.to(device)
        
        # Initialize face enhancer
        self.face_enhancer = None
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def load_model(self):
        """Load PIRender model from checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"PIRender checkpoint not found at {self.checkpoint_path}")
        
        # Import PIRender model architecture
        from src.facerender.pirender_arch import PIRenderArch
        model = PIRenderArch()
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def set_face_enhancer(self, enhancer_name):
        """Set face enhancer"""
        if enhancer_name is not None:
            self.face_enhancer = enhancer_generator_with_len(enhancer_name, self.device)
    
    def preprocess_image(self, image):
        """Preprocess image for PIRender model"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize to 256x256
        image = image.resize((256, 256), Image.LANCZOS)
        # Convert to tensor
        image = self.transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def preprocess_coeff(self, coeff):
        """Preprocess coefficients for PIRender model"""
        # Convert coefficients to the format expected by PIRender
        # This is a placeholder - adjust according to PIRender's requirements
        return torch.from_numpy(coeff).float().to(self.device)
    
    @torch.no_grad()
    def generate(self, coeff_path, source_image, result_dir, enhancer=None):
        """Generate video from coefficients"""
        # Set face enhancer if specified
        self.set_face_enhancer(enhancer)
        
        # Load source image
        source = self.preprocess_image(source_image)
        
        # Load coefficients
        coeffs = np.load(coeff_path)
        coeffs = self.preprocess_coeff(coeffs)
        
        # Generate frames
        frames = []
        for i in range(len(coeffs)):
            # Get current coefficient
            coeff = coeffs[i:i+1]
            
            # Generate frame
            with torch.cuda.amp.autocast():
                output = self.model(source, coeff)
            
            # Convert to image
            frame = (output[0].cpu().numpy().transpose(1, 2, 0) + 1) * 0.5
            frame = (frame * 255).astype(np.uint8)
            
            # Apply face enhancement if specified
            if self.face_enhancer is not None:
                frame = self.face_enhancer.enhance(frame)
            
            frames.append(frame)
        
        # Save video
        save_path = os.path.join(result_dir, 'result.mp4')
        save_video_with_watermark(frames, save_path, None)
        
        return save_path 
