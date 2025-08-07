#!/usr/bin/env python3
"""
Setup script for Python Code AI - Day 4
"""
import os
import subprocess
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'tokenizer', 'model', 'checkpoints']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install requirements")
        return False
    return True

def check_training_data():
    """Check if training data exists"""
    data_path = Path('data/train.txt')
    if data_path.exists():
        print(f"✓ Training data found: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"  File size: {len(content)} characters")
    else:
        print("⚠ No training data found at data/train.txt")
        print("  The training script will create sample data automatically")

def main():
    """Main setup function"""
    print("="*60)
    print("Python Code AI - Day 4 Setup")
    print("="*60)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        sys.exit(1)
    
    # Check training data
    print("\n3. Checking training data...")
    check_training_data()
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your training data to data/train.txt (optional)")
    print("2. Run: python train.py --epochs 10")
    print("3. After training: python generate.py --interactive")
    print("="*60)

if __name__ == "__main__":
    main()