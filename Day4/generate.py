import torch
import torch.nn.functional as F
from tokenizer.train_tokenizer import BPETokenizer
from model.transformer import CodeTransformer
from db.code_db import CodeDatabase

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.8):
    model.eval()
    
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            if generated.size(1) >= 256:
                break
                
            logits = model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, 1)
            
            if next_token_id.item() == tokenizer.token_to_id.get('<eos>', 2):
                break
            
            generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text

def load_model_and_tokenizer():
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer/tokenizer.pkl')
    
    # Load model
    checkpoint = torch.load('model/trained_model.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer

def main():
    model, tokenizer = load_model_and_tokenizer()
    db = CodeDatabase()
    
    while True:
        prompt = input("\nEnter your coding question (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        # Check if answer exists in database
        existing_answer = db.get_answer(prompt)
        if existing_answer:
            print(f"\n[From Database] {existing_answer}")
            continue
        
        # Generate new answer
        full_prompt = f"# Task: {prompt}\n"
        response = generate_response(model, tokenizer, full_prompt)
        
        # Extract just the code part
        if '\n' in response:
            code_part = '\n'.join(response.split('\n')[1:])
        else:
            code_part = response
        
        print(f"\n[Generated] {code_part}")
        
        # Save to database
        db.save_qa(prompt, code_part)

if __name__ == "__main__":
    main()