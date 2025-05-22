import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2

# List of available enhancers
enhancer_list = ['gfpgan', 'gpen']

class FaceEnhancer:
    def __init__(self, enhancer_name, device):
        self.device = device
        self.enhancer_name = enhancer_name
        self.model = self.load_enhancer()
        
    def load_enhancer(self):
        """Load the specified face enhancer model"""
        if self.enhancer_name == 'gfpgan':
            try:
                from gfpgan import GFPGANer
                model = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    device=self.device
                )
                return model
            except ImportError:
                print("GFPGAN not installed. Please install it with: pip install gfpgan")
                return None
        elif self.enhancer_name == 'gpen':
            try:
                from gpen import GPEN
                model = GPEN(
                    model_path='https://github.com/yangxy/GPEN/releases/download/v0.0.0/GPEN-BFR-512.pth',
                    device=self.device
                )
                return model
            except ImportError:
                print("GPEN not installed. Please install it with: pip install gpen")
                return None
        else:
            return None
    
    def enhance(self, img):
        """Enhance a single image"""
        if self.model is None:
            return img
            
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        # Convert to numpy array
        img = np.array(img)
        
        # Enhance image
        if self.enhancer_name == 'gfpgan':
            _, _, output = self.model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        elif self.enhancer_name == 'gpen':
            output = self.model.enhance(img)
        else:
            output = img
            
        return output

def enhancer_generator_with_len(enhancer_name, device):
    """Create a face enhancer instance"""
    return FaceEnhancer(enhancer_name, device)

def enhancer(images, method='gfpgan', bg_upsampler='realesrgan'):
    print('face enhancer....')
    if os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    # ------------------------ set up GFPGAN restorer ------------------------
    if  method == 'gfpgan':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    elif method == 'codeformer': # TODO:
        arch = 'CodeFormer'
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    else:
        raise ValueError(f'Wrong model version {method}.')


    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # determine model paths
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        model_path = os.path.join('checkpoints', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    restored_img = [] 
    for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        
        img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
        
        # restore faces and background if necessary
        cropped_faces, restored_faces, r_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True)
        
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        
        restored_img += [r_img]
       
    return restored_img

