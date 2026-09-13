# This script is used to test LLM models
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "Qwen/Qwen2.5-3B-Instruct"

# 1. Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Load the unquantized model strictly on your CPU
print("Loading model into RAM (this will take a moment)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,  # Safe for your 12GB RAM (torch_dtype is deprecated, use dtype instead)
    low_cpu_mem_usage=True,
    device_map={"": "cpu"},     # Forces CPU execution
)

# 3. Set up the text streamer for real-time word output
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 4. Initialize an empty chat history list
chat_history = []

print("\n=============================================")
print(" Qwen2.5 Interactive Chat Ready (CPU Mode) ")
print(" Type 'exit' or 'quit' to end the session. ")
print("=============================================\n")

# 5. Start the interactive loop
while True:
    try:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit commands
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        # Skip empty inputs
        if not user_input.strip():
            continue
            
        # Append user message to history
        chat_history.append({"role": "user", "content": user_input})
        
        # Format the text with chat templates (handles context)
        text = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
        
        # Generate response and stream it to terminal
        print("AI: ", end="", flush=True)
        generated_ids = model.generate(
            **model_inputs, 
            streamer=streamer, 
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract only the newly generated text to append to history
        # This strips out the input tokens so history stays clean
        input_length = model_inputs.input_ids.shape[1]
        ai_response_ids = generated_ids[0][input_length:]
        ai_response_text = tokenizer.decode(ai_response_ids, skip_special_tokens=True)
        
        # Append AI response to history for context in next turn
        chat_history.append({"role": "assistant", "content": ai_response_text})
        print() # Print a new line for spacing
        
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C
        print("\nSession interrupted. Goodbye!")
        break
