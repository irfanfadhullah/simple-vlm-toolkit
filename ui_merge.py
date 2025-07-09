## ui_merge.py
import os
import torch
import gradio as gr
from unsloth import FastLanguageModel

def merge_lora_adapter_to_model(base_model_name, adapter_path, output_dir, quantization="4bit", save_method="merged_4bit", progress=gr.Progress()):
    """
    Loads a base model, attaches LoRA weights, merges them, and saves the full model.
    """
    try:
        progress(0, desc="Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            load_in_4bit=True,
        )
        
        progress(0.4, desc="Loading and merging LoRA adapter...")
        # This will merge the adapter automatically
        model.load_adapter(adapter_path)
        model.merge_and_unload()
        
        progress(0.8, desc=f"Saving merged model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        if save_method == "gguf":
            model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q4_k_m")
            message = f"✅ GGUF model and tokenizer successfully saved to: {output_dir}"
        else: # Default to Hugging Face format
            model.save_pretrained(output_dir, tokenizer)
            message = f"✅ Merged model successfully saved to: {output_dir}"
        
        progress(1, desc="Done!")
        print(message)
        return message
        
    except Exception as e:
        error_msg = f"❌ Error during merging: {str(e)}"
        print(error_msg)
        return error_msg

def create_merge_interface():
    """
    Creates a Gradio interface for manually merging a LoRA adapter into a base model.
    """
    with gr.Blocks() as merge_interface:
        gr.Markdown("## Manual LoRA Adapter Merger")
        gr.Markdown("Use this tool if you trained a model with LoRA and want to merge it into a full model separately.")
        
        base_model_name = gr.Textbox(
            label="Base Model Name / Path", 
            value="unsloth/Llama-3-8B-Instruct-bnb-4bit",
            info="The original Hugging Face model ID you trained from."
        )
        adapter_path = gr.Textbox(
            label="Adapter Path", 
            placeholder="e.g., outputs/my-lora-model/lora_adapter",
            info="Path to the saved LoRA adapter weights."
        )
        output_dir = gr.Textbox(
            label="Output Directory", 
            placeholder="./merged_model_v1",
            info="Directory to save the final merged model."
        )
        save_method = gr.Radio(
            label="Saving Format",
            choices=["huggingface", "gguf"],
            value="huggingface",
            info="Choose 'huggingface' for general use or 'gguf' for llama.cpp."
        )
        
        merge_btn = gr.Button("Merge Adapter", variant="primary")
        merge_output = gr.Textbox(label="Merge Output", interactive=False, lines=3)
        
        merge_btn.click(
            merge_lora_adapter_to_model,
            inputs=[base_model_name, adapter_path, output_dir, gr.State("4bit"), save_method],
            outputs=merge_output
        )
    return merge_interface