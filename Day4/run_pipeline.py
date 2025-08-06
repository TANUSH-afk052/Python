import os
import sys
from utils import create_directories
from train import train_model
from generate import main as generate_main

def main():
    print("Setting up Code AI pipeline...")
    
    # Create directories
    create_directories()
    print("✓ Directories created")
    
    # Train model
    print("Starting training...")
    train_model()
    print("✓ Model trained and saved")
    
    # Start generation interface
    print("Starting generation interface...")
    generate_main()

if __name__ == "__main__":
    main()