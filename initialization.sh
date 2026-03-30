#!/bin/bash
# Initialization script for Handwriting Recognition System

# Update system packages
sudo apt-get update

# Install required packages
sudo apt-get install -y python3 python3-pip

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Print success message
echo "Initialization complete!"