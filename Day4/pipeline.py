#!/usr/bin/env python3
"""
Simple Python Code AI Pipeline
Easy training control with checkpoints
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils import create_directories, setup_logging
from train import train_model, find_latest_checkpoint

def setup_project():
    """Setup project directories"""
    dirs = [
        'data', 'model', 'tokenizer', 'checkpoints', 
        'logs', 'db', 'src', 'src/model', 'src/tokenizer'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    init_content = '# Package marker\n'
    for init_file in ['src/__init__.py', 'src/model/__init__.py', 'src/tokenizer/__init__.py']:
        if not Path(init_file).exists():
            Path(init_file).write_text(init_content)

def train_command(args):
    """Handle training command"""
    print(f"Starting training for {args.epochs} epochs...")
    
    if args.resume:
        checkpoint = find_latest_checkpoint('checkpoints')
        if checkpoint:
            print(f"Resuming from: {checkpoint}")
        else:
            print("No checkpoint found, starting fresh")
    
    try:
        train_model(epochs=args.epochs, resume=args.resume, checkpoint_path=args.checkpoint)
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted - progress saved!")
        print("Use: python pipeline.py train --resume")
        
    except Exception as e:
        print(f"Training error: {e}")

def status_command(args):
    """Show training status"""
    checkpoint_dir = Path('checkpoints')
    
    if not checkpoint_dir.exists():
        print("No training started yet")
        return
    
    # Find latest checkpoint
    latest = find_latest_checkpoint('checkpoints')
    if latest:
        import torch
        checkpoint = torch.load(latest, map_location='cpu')
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Latest checkpoint: Epoch {epoch}, Loss: {loss:.4f}")
        print(f"Checkpoint file: {latest}")
        
        # List all checkpoints
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            print(f"\nAvailable checkpoints: {len(checkpoints)}")
            for cp in sorted(checkpoints)[-5:]:  # Show last 5
                print(f"  {cp.name}")
    else:
        print("No checkpoints found")

def generate_command(args):
    """Handle generation command"""
    try:
        from generate import main as generate_main
        generate_main()
    except Exception as e:
        print(f"Generation error: {e}")
        print("Make sure the model is trained first")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Python Code AI - Simple Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=5, 
                             help='Number of epochs (default: 5)')
    train_parser.add_argument('--resume', action='store_true', 
                             help='Resume from latest checkpoint')
    train_parser.add_argument('--checkpoint', type=str, 
                             help='Specific checkpoint file to resume from')
    
    # Status command
    subparsers.add_parser('status', help='Show training status')
    
    # Generate command
    subparsers.add_parser('generate', help='Start generation interface')
    
    # Setup command
    subparsers.add_parser('setup', help='Setup project structure')
    
    args = parser.parse_args()
    
    # Setup project first
    setup_project()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'status':
        status_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'setup':
        print("Project setup completed!")
    else:
        # Default: show help and quick start
        parser.print_help()
        print("\nQuick Start:")
        print("1. python pipeline.py setup          # Setup project")
        print("2. python pipeline.py train --epochs 3   # Train for 3 epochs")
        print("3. python pipeline.py status         # Check progress")
        print("4. python pipeline.py train --resume --epochs 2  # Train 2 more")
        print("5. python pipeline.py generate       # Use the model")

if __name__ == "__main__":
    main()