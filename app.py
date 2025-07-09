## app.py
import gradio as gr
import os

from ui_dataset import create_dataset_maker_interface
from ui_train import create_train_interface
from ui_merge import create_merge_interface
from ui_chat import create_llm_chat_interface, create_vlm_chat_interface

def create_main_interface():
    """
    Creates the main Gradio interface with separate tabs for all functionalities.
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Simple VLM Toolkit") as main_interface:
        gr.Markdown(
            "# Simple VLM Toolkit\n"
            "An all-in-one interface for dataset creation, training, merging, and chatting with language and vision models."
        )

        with gr.Tabs():
            with gr.TabItem("VLM Chat"):
                create_vlm_chat_interface()

            with gr.TabItem("LLM Chat"):
                create_llm_chat_interface()

            with gr.TabItem("Train"):
                create_train_interface()

            with gr.TabItem("Merge Adapter"):
                create_merge_interface()
                
            with gr.TabItem("VLM Dataset Maker"):
                create_dataset_maker_interface()

    return main_interface

if __name__ == "__main__":
    interface = create_main_interface()
    # Use share=True to create a public link for easy sharing
    interface.launch(share=True)