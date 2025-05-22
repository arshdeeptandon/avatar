import os
import cv2
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle
import torch
from src.utils.croper import landmark_98_to_68
from src.utils.face_parsing import init_parser, get_face_mask
from src.utils.croper import get_final_mask

from torch.multiprocessing import Pool, Process, set_start_method

class KeypointExtractor():
    def __init__(self, device='cuda'):
        self.device = device
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                    flip_input=False, 
                                                    device=device)
        self.parser = init_parser(device=device)

    def extract_keypoint(self, images, name=None, info=True):
        """
        Extract keypoints from images
        Args:
            images: PIL Image or list of PIL Images
            name: path to save landmarks (optional)
            info: whether to show progress bar
        Returns:
            landmarks: numpy array of shape (N, 68, 2)
        """
        if isinstance(images, list):
            landmarks = []
            if info:
                i_range = tqdm(images,desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image)
                if np.mean(current_kp) == -1 and landmarks:
                    landmarks.append(landmarks[-1])
                else:
                    landmarks.append(current_kp[None])

            landmarks = np.concatenate(landmarks, 0)
            if name is not None:
                np.save(name, landmarks)
            return landmarks
        else:
            # Convert PIL Image to numpy array if needed
            if isinstance(images, Image.Image):
                images = np.array(images)
            
            # Convert to RGB if image is grayscale
            if len(images.shape) == 2:
                images = cv2.cvtColor(images, cv2.COLOR_GRAY2RGB)
            elif images.shape[2] == 4:  # If image has alpha channel
                images = images[:, :, :3]
            
            while True:
                try:
                    with torch.no_grad():
                        keypoints = self.detector.get_landmarks_from_image(images)[0]
                        break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        torch.cuda.empty_cache()
                    else:
                        images = cv2.resize(images, (0,0), fx=0.5, fy=0.5)
                        print('resize to', images.shape)
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)                    
                    break
            keypoints = keypoints.astype(np.float32)
            if name is not None:
                np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints

    def extract_keypoint_with_parsing(self, images, name=None, info=True):
        """
        Extract keypoints and masks from images
        Args:
            images: PIL Image or list of PIL Images
            name: path to save landmarks (optional)
            info: whether to show progress bar
        Returns:
            landmarks: numpy array of shape (N, 68, 2)
            masks: numpy array of shape (N, H, W)
        """
        if isinstance(images, list):
            landmarks = []
            masks = []
            if info:
                i_range = tqdm(images,desc='landmark Det:')
            else:
                i_range = images
                
            for image in i_range:
                current_kp, current_mask = self.extract_keypoint_with_parsing(image)
                if np.mean(current_kp) == -1 and landmarks:
                    landmarks.append(landmarks[-1])
                    masks.append(masks[-1])
                else:
                    landmarks.append(current_kp[None])
                    masks.append(current_mask[None])
                    
            landmarks = np.concatenate(landmarks, 0)
            masks = np.concatenate(masks, 0)
            if name is not None:
                np.save(name, landmarks)
                np.save(name.replace('.npy', '_mask.npy'), masks)
            return landmarks, masks
        else:
            # Convert PIL Image to numpy array if needed
            if isinstance(images, Image.Image):
                images = np.array(images)
            
            # Convert to RGB if image is grayscale
            if len(images.shape) == 2:
                images = cv2.cvtColor(images, cv2.COLOR_GRAY2RGB)
            elif images.shape[2] == 4:  # If image has alpha channel
                images = images[:, :, :3]
            
            while True:
                try:
                    with torch.no_grad():
                        keypoints = self.detector.get_landmarks_from_image(images)[0]
                        mask = get_face_mask(images, self.parser, normalize=False)
                        mask = get_final_mask(mask)
                        mask = cv2.resize(mask, (256, 256))
                        break
                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        torch.cuda.empty_cache()
                    else:
                        images = cv2.resize(images, (0,0), fx=0.5, fy=0.5)
                        print('resize to', images.shape)
                        
            keypoints = keypoints.astype(np.float32)
            return keypoints, mask

def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split('/')[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(
        images, 
        name=os.path.join(opt.output_dir, name[-2], name[-1])
    )

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    
    for ext in extensions:
        os.listdir(f'{opt.input_dir}')
        print(f'{opt.input_dir}/*.{ext}')
        filenames = sorted(glob.glob(f'{opt.input_dir}/*.{ext}'))
    print('Total number of videos:', len(filenames))
    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(filenames, args_list, device_ids))):
        None

