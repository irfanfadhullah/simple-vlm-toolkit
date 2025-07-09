## ui_chat.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer
from unsloth import FastLanguageModel, FastVisionModel
from peft import PeftModel
from PIL import Image
from threading import Thread
from qwen_vl_utils import process_vision_info # Make sure this utility is available for VLM

# Global state for loaded models
MODEL_STATE = {
    "model": None,
    "tokenizer": None,
    "processor": None,
    "model_type": None, # 'llm' or 'vlm'
}

# --- Model Loading Logic ---
def load_model(model_path, model_type, load_method, adapter_path, progress=gr.Progress()):
    """Unified function to load any model (LLM or VLM, full, or with adapter)."""
    global MODEL_STATE
    
    if not model_path:
        return "Error: Model Path / ID cannot be empty."
    if load_method == "Base Model + LoRA Adapter" and not adapter_path:
        return "Error: Adapter path is required for this load method."

    progress(0, desc="Configuring quantization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Unload previous model
    unload_model()

    try:
        progress(0.2, desc=f"Loading model: {model_path}...")
        
        # Determine the correct Unsloth class
        ModelClass = FastVisionModel if model_type == 'vlm' else FastLanguageModel

        # Case 1: Load a full model (merged or from HF hub)
        if load_method in ["Full Model from HF", "Merged Model from Local Folder"]:
            model, tokenizer = ModelClass.from_pretrained(
                model_name=model_path,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                device_map="auto",
            )
            MODEL_STATE["model"] = model

        # Case 2: Load a base model and attach a LoRA adapter
        elif load_method == "Base Model + LoRA Adapter":
            base_model, tokenizer = ModelClass.from_pretrained(
                model_name=model_path,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                device_map="auto",
            )
            progress(0.6, desc=f"Attaching adapter from {adapter_path}...")
            MODEL_STATE["model"] = PeftModel.from_pretrained(base_model, adapter_path)

        MODEL_STATE["tokenizer"] = tokenizer
        MODEL_STATE["model_type"] = model_type
        
        # Load processor only for VLM
        if model_type == 'vlm':
            progress(0.8, desc="Loading processor...")
            MODEL_STATE["processor"] = AutoProcessor.from_pretrained(model_path)
        else:
            MODEL_STATE["processor"] = None
        
        progress(1.0, desc="Model loaded successfully!")
        return f"✅ Success! Loaded {model_path} ({model_type})."

    except Exception as e:
        unload_model()
        return f"❌ Error loading model: {e}"

def unload_model():
    """Unloads the model and clears GPU memory."""
    global MODEL_STATE
    MODEL_STATE = {"model": None, "tokenizer": None, "processor": None, "model_type": None}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Model unloaded."

# --- Chat Generation Logic ---
def generate_chat_response(message, history, image, max_tokens, temp, top_p):
    """Handles chat generation for both LLM and VLM."""
    if not MODEL_STATE["model"]:
        yield "Please load a model first."
        return

    # Check if image is required but not provided
    if MODEL_STATE["model_type"] == 'vlm' and image is None:
        yield "This is a Vision model. Please upload an image."
        return

    tokenizer = MODEL_STATE["tokenizer"]
    model = MODEL_STATE["model"]
    processor = MODEL_STATE["processor"]
    
    # Prepare conversation history
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    
    # Add the current user message
    if MODEL_STATE["model_type"] == 'vlm':
        # VLM requires specific content format with image
        conversation.append({"role": "user", "content": [{"type": "text", "text": message}, {"type": "image", "image": image}]})
        text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation)
        inputs = processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to("cuda")
    else:
        # LLM just needs text
        conversation.append({"role": "user", "content": message})
        inputs = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        do_sample=True,
    )
    
    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    response = ""
    for new_text in streamer:
        response += new_text
        yield response

# --- UI Creation ---
def create_chat_ui(model_type):
    """Generic function to create a chat UI (either LLM or VLM)."""
    with gr.Blocks(elem_id=f"{model_type}_chat_interface") as interface:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"## {model_type.upper()} Model Loader")
                load_method = gr.Radio(
                    label="Load Method",
                    choices=["Full Model from HF", "Base Model + LoRA Adapter", "Merged Model from Local Folder"],
                    value="Full Model from HF",
                )
                model_path = gr.Textbox(label="Model Path / ID", placeholder="e.g., unsloth/Qwen2-1.5B-Instruct")
                adapter_path = gr.Textbox(label="LoRA Adapter Path (if applicable)", placeholder="e.g., outputs/my_lora_model")
                load_btn = gr.Button("Load Model", variant="primary")
                unload_btn = gr.Button("Unload Model")
                status_box = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=3):
                gr.Markdown(f"## {model_type.upper()} Chat")
                image_input = gr.Image(type="pil", label="Upload Image") if model_type == 'vlm' else gr.Textbox(visible=False)
                
                chatbot = gr.Chatbot(label=f"{model_type.upper()} Chatbot", height=500)
                msg = gr.Textbox(label="Your Message", placeholder="Type your message here...", container=False)
                
                with gr.Accordion("Generation Parameters", open=False):
                    max_tokens = gr.Slider(256, 4096, value=1024, label="Max New Tokens")
                    temp = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")
                
                clear_btn = gr.Button("Clear Chat")

        # Event Handlers
        load_btn.click(
            load_model,
            inputs=[model_path, gr.State(model_type), load_method, adapter_path],
            outputs=[status_box]
        )
        unload_btn.click(unload_model, outputs=[status_box])

        # Chat submission
        chat_inputs = [msg, chatbot, image_input, max_tokens, temp, top_p] if model_type == 'vlm' else [msg, chatbot, max_tokens, temp, top_p]
        msg.submit(generate_chat_response, inputs=chat_inputs, outputs=chatbot)
        msg.submit(lambda: "", inputs=[], outputs=[msg]) # Clear input textbox
        
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        
    return interface

def create_vlm_chat_interface():
    return create_chat_ui(model_type='vlm')

def create_llm_chat_interface():
    return create_chat_ui(model_type='llm')