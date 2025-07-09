# Simple VLM Toolkit

A user-friendly, all-in-one Gradio interface for interacting with, training, and managing both Vision Language Models (VLMs) and standard Language Models (LLMs). Powered by the high-speed `unsloth` library.

This toolkit provides a seamless workflow from dataset creation to fine-tuning and interactive chat, all within a single, easy-to-use web interface.

## ✨ Features

- **Dual Model Support**: Works with both Vision Language Models (VLM) and text-only Language Models (LLM)
- **Interactive Chat**: Separate, intuitive chat interfaces for VLM (with image upload) and LLM
- **Simplified Model Loading**: Easily load full models from Hugging Face, local folders, or a base model with a LoRA adapter
- **Flexible Training**:
    - Fine-tune models using LoRA for memory efficiency
    - Option to automatically merge the LoRA adapter into a full model after training
    - Dynamically populated model lists for VLM and LLM
- **VLM Dataset Creator**: A utility to process an Excel file and images into a VLM-ready dataset
- **Manual Adapter Merger**: A separate tool to merge a trained LoRA adapter into its base model at any time


## 🚀 Installation

**Prerequisites:**

- Python 3.9+
- An NVIDIA GPU with CUDA installed (required for `unsloth`)

**Steps:**

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/Simple-VLM-Toolkit.git
cd Simple-VLM-Toolkit
```

2. **Create and activate a virtual environment (recommended):**

```bash
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env
```

3. **Install the required packages:**
The `unsloth` library will automatically handle the installation of CUDA-specific libraries like bitsandbytes.

```bash
pip install -r requirements.txt
```

4. **Install the most suitable Unsloth version for your environment:**
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

## 🏃‍♀️ Running the Application

Once installation is complete, you can start the Gradio interface with a single command:

```bash
python app.py
```

This will launch a local web server. Open the URL provided in your terminal (usually http://127.0.0.1:7860) to access the toolkit.

## 📂 Project Structure

- `app.py`: The main entry point that launches the Gradio application
- `ui_*.py`: Python scripts that define the UI and logic for each tab (Chat, Train, etc.)
- `requirements.txt`: A list of all necessary Python packages
- `data/`: A placeholder directory for your datasets. See `data/README.md` for instructions on how to structure your files for the VLM Dataset Maker
- `outputs/`: This directory will be created automatically to save trained models and adapters. It is ignored by Git


## 📝 License

This project is lfree to use.
