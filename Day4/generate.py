import torch
import torch.nn.functional as F
from pathlib import Path
import logging

from src.tokenizer.train_tokenizer import BPETokenizer
from src.model.transformer import CodeTransformer
from src.db.python_code_db import PythonCodeDatabase
from src.utils import setup_device

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Enhanced code generator with better sampling strategies"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate_response(self, prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.9):
        """Generate response with advanced sampling"""
        try:
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt)
            if not input_ids:
                logger.warning("Empty prompt encoding")
                return ""
            
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            generated = input_ids
            
            with torch.no_grad():
                for _ in range(max_length):
                    if generated.size(1) >= self.model.config.max_seq_len:
                        break
                    
                    # Get logits
                    logits = self.model(generated)
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, 1)
                    
                    # Check for end token
                    if next_token_id.item() == self.tokenizer.token_to_id.get('<eos>', 2):
                        break
                    
                    generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)
            
            # Decode generated sequence
            generated_text = self.tokenizer.decode(generated[0].tolist())
            
            # Clean up the response
            return self._clean_response(generated_text, prompt)
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return f"Sorry, I encountered an error generating the response: {str(e)}"
    
    def _clean_response(self, generated_text, original_prompt):
        """Clean and format the generated response"""
        try:
            # Remove the original prompt from the response
            if original_prompt in generated_text:
                response = generated_text.replace(original_prompt, "").strip()
            else:
                response = generated_text.strip()
            
            # Clean up common artifacts
            response = response.replace('<pad>', '').replace('<unk>', '').replace('<bos>', '')
            
            # If response starts with newline, clean it
            response = response.lstrip('\n').rstrip()
            
            return response if response else "I couldn't generate a proper response for that query."
            
        except Exception as e:
            logger.error(f"Response cleaning failed: {str(e)}")
            return generated_text

def load_model_and_tokenizer():
    """Load trained model and tokenizer"""
    device = setup_device()
    
    # Load tokenizer
    tokenizer_path = Path('tokenizer/python_tokenizer.pkl')
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(tokenizer_path))
    logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    
    # Load model
    model_path = Path('model/python_code_model.pt')
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    checkpoint = torch.load(str(model_path), map_location=device)
    config = checkpoint['config']
    
    model = CodeTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded successfully. Best training loss: {checkpoint.get('best_loss', 'N/A')}")
    
    return model, tokenizer, device

def main():
    """Main generation interface"""
    try:
        logger.info("🚀 Loading Python Code AI...")
        
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer()
        
        # Initialize generator and database
        generator = CodeGenerator(model, tokenizer, device)
        db = PythonCodeDatabase()
        
        logger.info("✅ Python Code AI ready!")
        print("\n" + "="*60)
        print("🐍 PYTHON CODE AI - Ready to help with Python coding!")
        print("="*60)
        print("Commands:")
        print("  - Type your Python coding question")
        print("  - 'history' - Show recent Q&A history")
        print("  - 'search <keyword>' - Search previous answers")
        print("  - 'stats' - Show database statistics")
        print("  - 'quit' - Exit the program")
        print("="*60 + "\n")
        
        while True:
            try:
                prompt = input("🤔 Enter your Python coding question: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() == 'quit':
                    print("\n👋 Thanks for using Python Code AI! Goodbye!")
                    break
                
                elif prompt.lower() == 'history':
                    print("\n📚 Recent Q&A History:")
                    print("-" * 50)
                    history = db.get_recent_qa(10)
                    for i, (question, answer, timestamp) in enumerate(history, 1):
                        print(f"{i}. Q: {question[:60]}{'...' if len(question) > 60 else ''}")
                        print(f"   A: {answer[:60]}{'...' if len(answer) > 60 else ''}")
                        print(f"   Time: {timestamp}")
                        print()
                    continue
                
                elif prompt.lower().startswith('search '):
                    keyword = prompt[7:].strip()
                    if keyword:
                        print(f"\n🔍 Searching for '{keyword}':")
                        print("-" * 50)
                        results = db.search_questions(keyword)
                        for question, answer in results[:5]:
                            print(f"Q: {question}")
                            print(f"A: {answer}")
                            print()
                    else:
                        print("Please provide a search keyword.")
                    continue
                
                elif prompt.lower() == 'stats':
                    stats = db.get_stats()
                    print(f"\n📊 Database Statistics:")
                    print(f"Total Q&A pairs: {stats['total_pairs']}")
                    print(f"Generated answers: {stats['generated_count']}")
                    print(f"Database answers: {stats['database_count']}")
                    continue
                
                # Check database first
                print("\n🔍 Checking database...")
                existing_answer = db.get_answer(prompt)
                
                if existing_answer:
                    print(f"\n💾 [From Database]\n{existing_answer}")
                else:
                    print("🤖 Generating new answer...")
                    
                    # Generate new answer
                    full_prompt = f"# Task: {prompt}\n"
                    response = generator.generate_response(
                        full_prompt,
                        max_length=200,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9
                    )
                    
                    if response and response != full_prompt:
                        # Extract code part
                        if '\n' in response:
                            lines = response.split('\n')
                            code_lines = [line for line in lines[1:] if line.strip()]
                            code_part = '\n'.join(code_lines)
                        else:
                            code_part = response
                        
                        print(f"\n✨ [Generated]\n{code_part}")
                        
                        # Save to database
                        db.save_qa(prompt, code_part)
                        logger.info("Answer saved to database")
                    else:
                        print("\n❌ Sorry, I couldn't generate a proper response for that query.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using Python Code AI! Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"\n❌ An error occurred: {str(e)}")
                print("Please try again or type 'quit' to exit.")
    
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"❌ Failed to start Python Code AI: {str(e)}")
        print("Please ensure the model is trained first by running the training pipeline.")

if __name__ == "__main__":
    main()
