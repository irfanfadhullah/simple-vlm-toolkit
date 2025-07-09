## ui_train.py
import gradio as gr
import torch
from unsloth import FastLanguageModel, FastVisionModel, is_bf16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import os
from datetime import datetime

# --- Dataset Conversion Functions ---
def convert_to_vlm_conversation(sample):
    """Converts a VLM dataset sample into the conversation format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{sample.get('system_message', '')} {sample['messages']}".strip()},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant", "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
    }

def convert_to_llm_conversation(sample):
    """Converts a standard text dataset sample (e.g., Alpaca format) to conversation."""
    # This function assumes a dataset with 'instruction', 'input', and 'output' columns.
    # You might need to adjust it for your specific text dataset format.
    system = sample.get("system_message", "You are a helpful assistant.")
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")
    
    user_prompt = instruction
    if inp:
        user_prompt += f"\n\n{inp}"
        
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output},
        ]
    }

# --- Model Lists ---
VLM_MODELS = [
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",
    "unsloth/phi-3-vision-128k-instruct-bnb-4bit",
]
LLM_MODELS = [
    "unsloth/Llama-3-8B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/Qwen2-7B-Instruct-bnb-4bit",
    "unsloth/phi-3-medium-4k-instruct-bnb-4bit",
]

# --- Training Logic ---
class TrainingManager:
    def __init__(self):
        self.trainer = None
        self.is_training = False

    def get_save_directory(self, local_name=""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = local_name.strip() or f"model_{timestamp}"
        path = os.path.join("outputs", folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    def train_model(
        self, model_type, model_name, dataset_name, use_lora, max_length,
        batch_size, epochs, local_name, hf_repo, auto_merge, progress=gr.Progress()
    ):
        self.is_training = True
        progress(0, desc="[1/7] Initializing...")
        try:
            # Determine correct model class and conversion function
            ModelClass = FastVisionModel if model_type == 'vlm' else FastLanguageModel
            convert_fn = convert_to_vlm_conversation if model_type == 'vlm' else convert_to_llm_conversation
            
            # Load model and tokenizer
            progress(0.1, desc="[2/7] Loading model...")
            model, tokenizer = ModelClass.from_pretrained(
                model_name=model_name,
                load_in_4bit=True,
                use_gradient_checkpointing="unsloth",
            )
            
            if use_lora:
                progress(0.2, desc="[3/7] Configuring LoRA...")
                model = ModelClass.get_peft_model(
                    model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
                    # VLM-specific finetuning options
                    finetune_vision_layers=True if model_type == 'vlm' else None,
                )
            
            # Load and prepare dataset
            progress(0.3, desc="[4/7] Loading and processing dataset...")
            dataset = load_dataset(dataset_name, split="train")
            # For demonstration, we'll cap the dataset size
            if len(dataset) > 500:
                dataset = dataset.select(range(500))
            
            # The formatting function needs to be defined to handle the dict structure
            def formatting_func(sample):
                return convert_fn(sample)["messages"]

            output_dir = self.get_save_directory(local_name)
            
            progress(0.4, desc="[5/7] Setting up trainer...")
            self.trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                formatting_func=formatting_func,
                max_seq_length=max_length,
                args=SFTConfig(
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    num_train_epochs=epochs,
                    learning_rate=2e-4,
                    fp16=not is_bf16_supported(),
                    bf16=is_bf16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=output_dir,
                    report_to="none",
                ),
            )
            
            progress(0.5, desc="[6/7] Starting training...")
            train_stats = self.trainer.train()
            
            final_message = f"✅ Training complete!\nStats: {train_stats}\n"
            
            if use_lora:
                adapter_path = os.path.join(output_dir, "lora_adapter")
                model.save_pretrained(adapter_path)
                final_message += f"LoRA adapter saved to: {adapter_path}\n"
                
                if auto_merge:
                    progress(0.8, desc="[7/7] Merging adapter...")
                    merged_path = os.path.join(output_dir, "merged_model")
                    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_4bit")
                    final_message += f"✅ Model merged and saved to: {merged_path}\n"
                    
                    if hf_repo:
                        progress(0.9, desc=f"Pushing to Hugging Face Hub: {hf_repo}")
                        model.push_to_hub_merged(hf_repo, tokenizer, save_method="merged_4bit", token=True)
                        final_message += f"✅ Pushed merged model to Hugging Face Hub!"
                elif hf_repo:
                     progress(0.9, desc=f"Pushing adapter to Hugging Face Hub: {hf_repo}")
                     model.push_to_hub(hf_repo, tokenizer, save_method="lora", token=True)
                     final_message += f"✅ Pushed adapter to Hugging Face Hub!"
            
            else: # Full finetuning
                final_message += f"Full model saved in checkpoints at {output_dir}"

            self.is_training = False
            progress(1.0, desc="Done!")
            return final_message
            
        except Exception as e:
            self.is_training = False
            return f"❌ Error during training: {e}"

# --- UI Creation ---
def create_train_interface():
    manager = TrainingManager()
    
    with gr.Blocks() as train_interface:
        gr.Markdown("# Model Training Interface")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Model & Data Configuration")
                model_type = gr.Radio(["vlm", "llm"], label="Model Type", value="vlm")
                model_name = gr.Dropdown(label="Select a Model", choices=VLM_MODELS, value=VLM_MODELS[0])
                dataset_name = gr.Textbox(label="Dataset on Hugging Face Hub", placeholder="e.g., username/my_dataset")
                
                gr.Markdown("### 2. Training Parameters")
                use_lora = gr.Checkbox(label="Use LoRA Fine-tuning", value=True)
                with gr.Row():
                    epochs = gr.Slider(1, 10, value=1, step=1, label="Epochs")
                    batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                max_length = gr.Slider(1024, 8192, value=2048, step=256, label="Max Sequence Length")

                gr.Markdown("### 3. Saving & Uploading")
                with gr.Row():
                    local_name = gr.Textbox(label="Local Save Name (optional)", placeholder="my-finetuned-model")
                    hf_repo = gr.Textbox(label="HF Repo to Upload (optional)", placeholder="username/my-hf-repo")
                auto_merge = gr.Checkbox(label="Auto-merge LoRA adapter after training?", value=True, info="If checked, creates a full model ready for inference.")

                train_btn = gr.Button("Start Training", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Training Log")
                training_output = gr.Textbox(label="Status & Output", interactive=False, lines=25)

        # Dynamic UI updates
        def update_model_list(m_type):
            return gr.Dropdown(choices=VLM_MODELS if m_type == 'vlm' else LLM_MODELS, 
                               value=VLM_MODELS[0] if m_type == 'vlm' else LLM_MODELS[0])
        
        model_type.change(update_model_list, inputs=model_type, outputs=model_name)

        # Button click handler
        train_btn.click(
            manager.train_model,
            inputs=[model_type, model_name, dataset_name, use_lora, max_length, 
                    batch_size, epochs, local_name, hf_repo, auto_merge],
            outputs=training_output
        )

    return train_interface