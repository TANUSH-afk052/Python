import os
import torch
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from local files
from config import ModelConfig
from transformer import CodeTransformer
from train_tokenizer import BPETokenizer

def setup_device():
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def load_model_and_tokenizer():
    """Load trained model and tokenizer"""
    device = setup_device()
    
    # Paths
    model_path = "checkpoints/checkpoint_epoch_10.pt"
    tokenizer_path = "tokenizer/python_tokenizer.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please train the model first.")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    
    # Load model (fix for PyTorch >= 2.6)
    logger.info("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load model config (optional fallback)
    config = checkpoint.get('config', None)
    if config is None:
        class DummyConfig:
            vocab_size = len(tokenizer.vocab)
            d_model = 256
            nhead = 8
            num_layers = 6
            max_seq_len = 256
            pad_token_id = 0
            bos_token_id = 1
        config = DummyConfig()

    # Initialize model
    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded with {model.get_num_params():,} parameters")
    
    return model, tokenizer, device, config


def generate_code(model, tokenizer, device, config, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
    """Generate code based on prompt"""
    # Encode prompt
    try:
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            logger.warning("Empty encoding for prompt, using default start token")
            input_ids = [config.bos_token_id]
    except Exception as e:
        logger.warning(f"Encoding failed: {e}, using default start token")
        input_ids = [config.bos_token_id]
    
    # Convert to tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    logger.info(f"Generating with prompt: '{prompt}' (length: {len(input_ids)})")
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=config.pad_token_id
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    
    return generated_text

def interactive_mode():
    """Interactive generation mode"""
    print("\n" + "="*60)
    print("Python Code AI - Interactive Generation")
    print("="*60)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help message")
    print("  'params' - Show current generation parameters")
    print("  'set temp <value>' - Set temperature (0.1-2.0)")
    print("  'set length <value>' - Set max generation length")
    print("="*60)
    
    try:
        model, tokenizer, device, config = load_model_and_tokenizer()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Default generation parameters
    temperature = 0.8
    max_length = 100
    top_k = 50
    top_p = 0.9
    
    while True:
        try:
            prompt = input("\nEnter your prompt (or command): ").strip()
            
            if not prompt:
                continue
            
            # Handle commands
            if prompt.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif prompt.lower() == 'help':
                print("\nCommands:")
                print("  'quit' or 'exit' - Exit the program")
                print("  'help' - Show this help message")
                print("  'params' - Show current generation parameters")
                print("  'set temp <value>' - Set temperature (0.1-2.0)")
                print("  'set length <value>' - Set max generation length")
                continue
            elif prompt.lower() == 'params':
                print(f"\nCurrent parameters:")
                print(f"  Temperature: {temperature}")
                print(f"  Max length: {max_length}")
                print(f"  Top-k: {top_k}")
                print(f"  Top-p: {top_p}")
                continue
            elif prompt.lower().startswith('set temp'):
                try:
                    temp_value = float(prompt.split()[-1])
                    if 0.1 <= temp_value <= 2.0:
                        temperature = temp_value
                        print(f"Temperature set to {temperature}")
                    else:
                        print("Temperature must be between 0.1 and 2.0")
                except (ValueError, IndexError):
                    print("Invalid temperature value. Usage: set temp <value>")
                continue
            elif prompt.lower().startswith('set length'):
                try:
                    length_value = int(prompt.split()[-1])
                    if 10 <= length_value <= 500:
                        max_length = length_value
                        print(f"Max length set to {max_length}")
                    else:
                        print("Max length must be between 10 and 500")
                except (ValueError, IndexError):
                    print("Invalid length value. Usage: set length <value>")
                continue
            
            # Generate code
            print(f"\nGenerating...")
            generated = generate_code(
                model, tokenizer, device, config,
                prompt, max_length, temperature, top_k, top_p
            )
            
            print(f"\nGenerated:")
            print("-" * 40)
            print(generated)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")
            logger.error(f"Generation error: {e}")

def single_generation(prompt, max_length, temperature, top_k, top_p):
    """Single generation mode"""
    try:
        model, tokenizer, device, config = load_model_and_tokenizer()
        
        generated = generate_code(
            model, tokenizer, device, config,
            prompt, max_length, temperature, top_k, top_p
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated:")
        print("-" * 40)
        print(generated)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Generation error: {e}")

def main():
    """Main generation interface"""
    parser = argparse.ArgumentParser(description='Generate Python code using trained AI model')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Generation temperature (0.1-2.0)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or args.prompt is None:
        interactive_mode()
    else:
        single_generation(args.prompt, args.max_length, args.temperature, args.top_k, args.top_p)

if __name__ == "__main__":
    main()