# Talking Face Avatar - GPU Optimized

This repository contains an optimized version of the Talking Face Avatar project, specifically designed for GPU acceleration and AWS SageMaker deployment. The code has been optimized for better performance and memory management.

## Features

- GPU-accelerated face animation generation
- Optimized memory management
- Mixed precision training support
- Dynamic batch processing
- AWS SageMaker deployment ready
- Enhanced face detection and processing

## Prerequisites

- Python 3.8+
- CUDA 11.1+ (for GPU support)
- FFmpeg
- NVIDIA GPU with at least 8GB VRAM (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Talking_Face_Avatar_GPU.git
cd Talking_Face_Avatar_GPU
```

### 2. Create Conda Environment

```bash
conda create -n sadtalker_gpu python=3.8
conda activate sadtalker_gpu
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install FFmpeg
conda install ffmpeg

# Install other dependencies
pip install -r requirements.txt
```

### 4. Download Model Weights

Download the required model weights from the original repository and place them in the `checkpoints` directory:

```bash
mkdir -p checkpoints
# Download the following files to checkpoints/:
# - epoch_20.pth
# - auido2pose_00140-model.pth
# - auido2exp_00300-model.pth
# - facevid2vid_00189-model.pth.tar
# - mapping_00229-model.pth.tar
# - shape_predictor_68_face_landmarks.dat
```

## Usage

### Basic Usage

```bash
python inference.py \
    --driven_audio examples/driven_audio/RD_Radio34_002.wav \
    --source_image examples/source_image/happy1.png \
    --enhancer gfpgan \
    --preprocess full
```

### Optimized Version

For better performance, use the optimized version:

```bash
python inference_optimized.py \
    --driven_audio examples/driven_audio/RD_Radio34_002.wav \
    --source_image examples/source_image/happy1.png \
    --enhancer gfpgan \
    --preprocess full \
    --batch_size 8
```

### Command Line Arguments

- `--driven_audio`: Path to the input audio file
- `--source_image`: Path to the source face image
- `--enhancer`: Face enhancement method (gfpgan, gpen, or none)
- `--preprocess`: Preprocessing method (crop, resize, or full)
- `--batch_size`: Batch size for processing (default: 8 for GPUs >10GB, 4 for smaller GPUs)
- `--expression_scale`: Expression scale factor (default: 1.0)
- `--pose_style`: Pose style (default: 0)
- `--result_dir`: Output directory (default: ./results)

## AWS SageMaker Deployment

### 1. Create a SageMaker Notebook Instance

1. Go to AWS SageMaker Console
2. Create a new notebook instance with:
   - Instance type: ml.g4dn.xlarge (or larger for better performance)
   - Platform: conda_python3
   - Volume size: 50GB (minimum)

### 2. Clone and Setup

In your SageMaker notebook:

```python
# Clone the repository
!git clone https://github.com/yourusername/Talking_Face_Avatar_GPU.git
%cd Talking_Face_Avatar_GPU

# Create and activate conda environment
!conda create -n sadtalker_gpu python=3.8 -y
!conda activate sadtalker_gpu

# Install dependencies
!conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
!conda install ffmpeg -y
!pip install -r requirements.txt
```

### 3. Download Model Weights

Create a notebook cell to download the model weights:

```python
import os
import gdown

# Create checkpoints directory
os.makedirs('checkpoints', exist_ok=True)

# Download model weights
# Add your download links here
```

### 4. Run Inference

Create a notebook cell for inference:

```python
import subprocess

def run_inference(audio_path, image_path, output_dir='results'):
    cmd = f"""
    python inference_optimized.py \
        --driven_audio {audio_path} \
        --source_image {image_path} \
        --enhancer gfpgan \
        --preprocess full \
        --batch_size 8 \
        --result_dir {output_dir}
    """
    subprocess.run(cmd, shell=True, check=True)
```

## Performance Optimization Tips

1. **GPU Memory Management**:
   - Use `--batch_size` appropriate for your GPU memory
   - Monitor GPU memory usage with `nvidia-smi`
   - Consider using gradient checkpointing for large models

2. **Processing Speed**:
   - Use the optimized version (`inference_optimized.py`)
   - Enable mixed precision training
   - Use appropriate batch size for your GPU

3. **Quality vs Speed Trade-off**:
   - Use `--preprocess full` for best quality
   - Use `--preprocess crop` for faster processing
   - Adjust `--expression_scale` for different expression intensities

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient checkpointing
   - Clear GPU cache between runs

2. **Face Detection Issues**:
   - Ensure source image has clear face
   - Try different preprocessing methods
   - Check image resolution

3. **Installation Issues**:
   - Ensure CUDA version matches PyTorch
   - Use exact versions specified in requirements
   - Check GPU compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Talking Face Avatar project
- GFPGAN for face enhancement
- All other open-source projects used in this repository

<div align="center">

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/sabahesaraY)


 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb) &nbsp; 


<b>Talking Face Avatar:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; single portrait image From Leonardo.ai API 🙎‍♂️ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; audio From ElevenLabs TTS API 🎤 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; =  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; talking head video 🎞.</b>

<br>

</div>

## Leonardo.ai 

Go To [Leonardo.Ai](https://app.leonardo.ai/ai-generations) And Enter your Prompt And Negative Prompts To Generate Artistic Images

Here Some Recources :[Leonardo.ai Youtube Video](https://www.youtube.com/watch?v=36rSjS5hV4Y)
                     [Leonardo.ai Youtube Video Toutorial](https://www.youtube.com/watch?v=XW7CyTPd0aI&list=PL3qnMcmlvPBHvhxHL2wVHSzN870Y-wWVa)
                     
or you can use APIs [Leonardo.Ai API Guide](https://docs.leonardo.ai/docs) 

<table class="center">
<tr>
  <td style="text-align:center;"><b>Leonardo.ai Image Generation</b></td>
  <td style="text-align:center;"><b>Leonardo.ai Image Generation</b></td>
   <td style="text-align:center;"><b>Leonardo.ai Image Generation</b></td>
</tr>
  
<tr>
<td>
<img src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/12ffc93b-79f5-4cf0-a14d-f58cc050cc16" width="300px";height:"400px">

</td>
<td>
 <img src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/ef923464-1033-45ba-a067-2a21afbae8fa" width="300px";height:"400px">

</td>
<td>
 <img src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/cfb315c5-35f8-4a76-945d-eca0393825b0" width="300px";height:"400px">


</td>

</tr>
</table>


## ElevenLabs 

Go To [Eleven Labs](https://beta.elevenlabs.io/) And Enter your Text And Generate Beautiful Audios With Diffrent Pitchs and Speeckers. ElvenLabs also is Multilingual 

Here Some Recources :[ElevenLabs Youtube Video](https://www.youtube.com/watch?v=CYGwfHWiSyU)

or you can use APIs [ElevenLabs API Guide](https://api.elevenlabs.io/docs) 

[ElevenLabs Python Repo](https://github.com/lugia19/elevenlabslib)


<table class="center">
<tr>
  <td style="text-align:center;"><b>Eleven Labs TTS</b></td>
  <td style="text-align:center;"><b>Eleven Labs TTS</b></td>
   <td style="text-align:center;"><b>Eleven Labs TTS</b></td>
</tr>
  
<tr>
<td>


https://github.com/saba99/Talking_Face_Avatar/assets/33378412/bd68137d-2e67-41df-a1df-4162db170ff8


</td>
<td>

https://github.com/saba99/Talking_Face_Avatar/assets/33378412/f622369b-9e69-492d-975b-685671c663c1

</td>
<td>

https://github.com/saba99/Talking_Face_Avatar/assets/33378412/1f78eb67-cc76-4c9a-8664-a28f3f795bee

</td> 

 
</tr>


</table>




## 🔥 Highlight


-🔥 Scroll To left and Right To See All Videos


| video 1 + enhancer(GFPGAN )                | video 2      |  video 3 |
|:--------------------: |:--------------------: | :----: |
| <video src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/7879fcc4-fe3c-473a-86d8-23c736dd4a65.mp4" type="video/mp4"> </video> | <video  src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/a7eb4df8-1789-412d-97da-8f6b361a72a2.mp4" type="video/mp4"> </video>  | <video  src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/b20102bd-852a-4091-87da-18dba720bc93.mp4" type="video/mp4"> </video>

| video 4                | video 5     |  video 6 |
|:--------------------: |:--------------------: | :----: |
| <video src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/91be247e-b1e4-4206-aa95-4eca938597b2.mp4" type="video/mp4"> </video> | <video  src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/18c7fc59-00f9-4118-b88b-545d3ea87342.mp4" type="video/mp4"> </video>  | <video  src="https://github.com/saba99/Talking_Face_Avatar/assets/33378412/f0bc6062-1a81-44f7-b6bc-747df6c06568.mp4" type="video/mp4"> </video>

- 🔥 Several new mode, eg, `still mode`, `reference mode`, `resize mode` are online for better and custom applications.

## Our Diagram Approach


![you_doodle_pro_2023-05-19t16_25_09z](https://github.com/saba99/Talking_Face_Avatar/assets/33378412/a516b0fb-7ab5-4fd3-b78c-79808e8eec6e)



### Linux:

1. Installing [anaconda](https://www.anaconda.com/), python and git.

2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/saba99/Talking_Face_Avatar.git

  cd SadTalker 

  conda create -n sadtalker python=3.8

  conda activate sadtalker

  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

  conda install ffmpeg

  pip install -r requirements.txt

  ### tts is optional for gradio demo. 
  ### pip install TTS

  ```  
### UI + API:
  look at index.html 

![Screenshot (4899)](https://github.com/saba99/Talking_Face_Avatar/assets/33378412/f922a6e8-38da-4b2a-b339-7a689dc33bc9) 


## 📥 2. Download Trained Models.

You can run the following script to put all the models in the right place.




```bash
bash scripts/download_models.sh
```

<details><summary>Model Details</summary>

The final folder will be shown as:

<img width="331" alt="image" src="https://user-images.githubusercontent.com/4397546/232511411-4ca75cbf-a434-48c5-9ae0-9009e8316484.png">


Model explains:

| Model | Description
| :--- | :----------
|checkpoints/auido2exp_00300-model.pth | Pre-trained ExpNet in Sadtalker.
|checkpoints/auido2pose_00140-model.pth | Pre-trained PoseVAE in Sadtalker.
|checkpoints/mapping_00229-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/mapping_00109-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/facevid2vid_00189-model.pth.tar | Pre-trained face-vid2vid model from [the reappearance of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis).
|checkpoints/epoch_20.pth | Pre-trained 3DMM extractor in [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction).
|checkpoints/wav2lip.pth | Highly accurate lip-sync model in [Wav2lip](https://github.com/Rudrabha/Wav2Lip).
|checkpoints/shape_predictor_68_face_landmarks.dat | Face landmark model used in [dilb](http://dlib.net/). 
|checkpoints/BFM | 3DMM library file.  
|checkpoints/hub | Face detection models used in [face alignment](https://github.com/1adrianb/face-alignment).
|gfpgan/weights | Face detection and enhanced models used in `facexlib` and `gfpgan`.


</details>

## 🔮 3. Quick Start ([Best Practice](docs/best_practice.md)).

### WebUI Demos:

 [SDWebUI-Colab](https://colab.research.google.com/github/camenduru/stable-diffusion-webui-colab/blob/main/video/stable/stable_diffusion_1_5_video_webui_colab.ipynb) | [Colab](https://colab.research.google.com/github/Winfredy/SadTalker/blob/main/quick_demo.ipynb)

```bash
## you need manually install TTS(https://github.com/coqui-ai/TTS) via `pip install tts` in advanced.
python app.py
```

### Manually usages:

##### Animating a portrait image from default config:
```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --enhancer gfpgan 
```
The results will be saved in `results/$SOME_TIMESTAMP/*.mp4`.

##### Full body/image Generation:

Using `--still` to generate a natural full body video. You can add `enhancer` to improve the quality of the generated video. 

```bash
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --result_dir <a file to store results> \
                    --still \
                    --preprocess full \
                    --enhancer gfpgan 
```


[![Star History Chart](https://api.star-history.com/svg?repos=yazdi9/Talking_Face_Avatar&type=Date)](https://star-history.com/#yazdi9/Talking_Face_Avatar&Date)

