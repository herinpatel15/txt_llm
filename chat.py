from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chat():
    model_path = "./my_train_bot"  # Ensure this directory has both model and tokenizer files
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    print("Chatbot is ready! (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        outputs = model.generate(
            input_ids, 
            max_length=150, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Bot:", response)

if __name__ == "__main__":
    chat()
