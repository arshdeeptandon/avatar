import os
import torch
import numpy as np
import cv2
import argparse
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.utils.init_path import init_path
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.videoio import save_video_with_watermark
import warnings
warnings.filterwarnings('ignore')

def main(args):
    # Initialize paths
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, os.path.splitext(os.path.basename(pic_path))[0])
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    if device == 'cuda':
        # Enable TF32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        # Set GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        # Enable memory efficient attention
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # Enable mixed precision
        torch.backends.cuda.enable_flash_sdp(True)

    # Initialize models
    print('Initializing models...')
    
    # Initialize preprocess model
    print('Loading preprocess model...')
    preprocess_model = CropAndExtract(init_path(args), device)
    preprocess_model.net_recon.eval()  # Set to eval mode for inference
    
    # Initialize audio2coeff model
    print('Loading audio2coeff model...')
    audio_to_coeff = Audio2Coeff(init_path(args), device)
    
    # Initialize face renderer
    print('Loading face renderer...')
    if args.facerender == 'pirender':
        from src.facerender.pirender import PIRender
        animate_from_coeff = PIRender(init_path(args), device)
    else:  # default to facevid2vid
        animate_from_coeff = AnimateFromCoeff(init_path(args), device)
    
    # Process source image
    print('Processing source image...')
    first_frame_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, save_dir, args.preprocess, source_image_flag=True
    )
    
    # Process audio
    print('Processing audio...')
    with torch.cuda.amp.autocast():
        batch = get_data(first_frame_coeff_path, audio_path, device, None, still=args.still)
        coeff_path = audio_to_coeff.generate(batch, save_dir, args.pose_style, None)
    
    # Generate video
    print('Generating video...')
    with torch.cuda.amp.autocast():
        video_path = animate_from_coeff.generate(
            coeff_path, 
            crop_pic_path, 
            save_dir, 
            args.enhancer,
            background_enhancer=args.background_enhancer,
            preprocess=args.preprocess,
            still=args.still
        )
    
    # Clean up GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(f'Video generated successfully: {video_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code for talking face generation')
    
    # Add arguments
    parser.add_argument('--driven_audio', type=str, help='path to driven audio')
    parser.add_argument('--source_image', type=str, help='path to source image')
    parser.add_argument('--result_dir', type=str, default='./results', help='path to output')
    parser.add_argument('--enhancer', type=str, default=None, choices=enhancer_list, help='face enhancer, [gfpgan, gpen]')
    parser.add_argument('--background_enhancer', type=str, default=None, choices=enhancer_list, help='background enhancer, [gfpgan, gpen]')
    parser.add_argument('--cpu', action='store_true', help='use cpu inference')
    parser.add_argument('--preprocess', type=str, default='full', choices=['crop', 'resize', 'full'], help='preprocess, [crop, resize, full]')
    parser.add_argument('--still', action='store_true', help='still mode (no head motion)')
    parser.add_argument('--pose_style', type=int, default=0, help='input pose style from [0-23]')
    parser.add_argument('--facerender', type=str, default='facevid2vid',
                      choices=['facevid2vid', 'pirender'],
                      help='Face renderer to use (facevid2vid or pirender)')
    
    args = parser.parse_args()
    main(args) 
