#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download PIRender checkpoint
echo "Downloading PIRender checkpoint..."
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7' -O checkpoints/epoch_00190_iteration_000400000_checkpoint.pt

echo "Download complete!" 
