import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cache model for faster reload and efficient memory management
@st.cache_resource
def load_model():
    """
    Load and cache the Qwen2.5-1.5B-Instruct model and tokenizer.
    The model is optimized for inference with compilation.
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate dtype and device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for optimized memory usage
        device_map="cuda"  # Ensure the model is loaded on GPU
    )
    
    # Compile model for performance improvements
    model = torch.compile(model)  
    
    # Initialize chat messages
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    ]
    
    return model, tokenizer, messages

# Load the model, tokenizer, and initial messages
model, tokenizer, messages = load_model()

# Streamlit UI setup
st.title("Qwen2.5-1.5B-Instruct Chatbot")

def chat(prompt):
    """
    Generate a response using the model based on user input.
    """
    messages.append({"role": "user", "content": prompt})
    
    # Convert conversation to the model-specific chat format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize input and move it to GPU
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Generate model response with inference mode enabled for efficiency
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,  # Limit response length
            do_sample=True,  # Enable sampling for varied responses
            temperature=0.7,  # Control randomness in generation
            repetition_penalty=1.1  # Reduce repetition in responses
        )
    
    # Extract generated response tokens excluding input tokens
    output = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
    ]
    
    # Decode output tokens into human-readable text
    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    
    # Extract assistant's response and clean it up
    response_message = response.split("Assistant:")[-1].strip()
    
    return response_message

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate assistant response
    response = chat(prompt)
    
    # Display assistant response in chat
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Save response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
