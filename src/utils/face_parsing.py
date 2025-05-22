import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import warnings

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        self.n_classes = n_classes
        # Load pretrained model
        self.load_state_dict(torch.load(os.path.join('checkpoints', 'face_parsing.pth'), map_location='cpu'))
        self.eval()

    def forward(self, x):
        # Forward pass
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        x = self.backbone(x)
        x = self.head(x)
        return x

class SimpleFaceParser:
    """A simple face parser that uses landmarks to create a face mask"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def to(self, device):
        return self
    
    def eval(self):
        return self

def init_parser(device='cuda'):
    """Initialize face parsing model with fallback to simple parser"""
    try:
        model_path = os.path.join('checkpoints', 'face_parsing.pth')
        if not os.path.exists(model_path):
            warnings.warn("Face parsing model not found. Using simple landmark-based parser.")
            return SimpleFaceParser()
        
        parser = BiSeNet(n_classes=19)
        parser = parser.to(device)
        parser.eval()
        return parser
    except Exception as e:
        warnings.warn(f"Failed to load face parsing model: {str(e)}. Using simple landmark-based parser.")
        return SimpleFaceParser()

def get_face_mask(image, parser, normalize=True):
    """Get face mask using face parsing model or landmarks"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if isinstance(parser, SimpleFaceParser):
        # For simple parser, return a basic face mask
        mask = np.ones((512, 512), dtype=np.float32)
        return mask
    
    # Preprocess image
    image = parser.transform(image).unsqueeze(0)
    image = image.to(next(parser.parameters()).device)
    
    with torch.no_grad():
        output = parser(image)[0]
        output = F.softmax(output, dim=1)
        mask = output.argmax(dim=1).squeeze().cpu().numpy()
    
    # Create binary mask (face region = 1, background = 0)
    face_mask = np.zeros_like(mask)
    face_mask[mask > 0] = 1  # All face parts are marked as 1
    
    if normalize:
        face_mask = face_mask.astype(np.float32)
    
    return face_mask

def get_face_mask_with_landmarks(image, landmarks, parser, normalize=True):
    """Get face mask using both face parsing and landmarks"""
    if isinstance(parser, SimpleFaceParser):
        # For simple parser, use only landmarks
        h, w = 512, 512
        landmark_mask = np.zeros((h, w), dtype=np.float32)
        
        # Convert landmarks to numpy array if needed
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.cpu().numpy()
        
        # Draw convex hull of landmarks
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillConvexPoly(landmark_mask, hull, 1)
        return landmark_mask
    
    # Get parsing mask
    parsing_mask = get_face_mask(image, parser, normalize=False)
    
    # Create mask from landmarks
    h, w = parsing_mask.shape
    landmark_mask = np.zeros((h, w), dtype=np.float32)
    
    # Convert landmarks to numpy array if needed
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
    
    # Draw convex hull of landmarks
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(landmark_mask, hull, 1)
    
    # Combine masks
    combined_mask = np.logical_and(parsing_mask, landmark_mask).astype(np.float32)
    
    if normalize:
        combined_mask = combined_mask.astype(np.float32)
    
    return combined_mask 
