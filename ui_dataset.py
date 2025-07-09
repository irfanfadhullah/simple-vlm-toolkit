## ui_dataset.py
import os
import json
import random
import pandas as pd
from PIL import Image as PILImage
from datasets import load_dataset, Features, Value, Image
import gradio as gr

def run_simple_dataset(
    excel_path: str, image_dir: str, resized_dir: str, instruction_path: str,
    question_path: str, summary_path: str, caption_path: str,
    output_jsonl: str, hf_repo_id: str
) -> str:
    # ... (code is identical to your original dataset_maker.py) ...
    """
    Process a simple orthodontic VLM dataset:
      1) Resizes images to 512x512
      2) Generates a JSONL file with system_message, messages, answer, and image fields
      3) Uploads to Hugging Face Hub
    """
    try:
        os.makedirs(resized_dir, exist_ok=True)

        # Load data and text resources
        df = pd.read_excel(excel_path, engine='openpyxl')
        instruction_str = open(instruction_path, encoding='utf-8').read().strip()
        questions = [l.strip() for l in open(question_path, 'r', encoding='utf-8') if l.strip()]
        summaries = [l.strip() for l in open(summary_path, 'r', encoding='utf-8') if l.strip()]
        captions = [l.strip() for l in open(caption_path, 'r', encoding='utf-8') if l.strip()]
        assert len(questions) == len(summaries) == len(captions), \
            "question.txt, summary.txt and caption.txt must all have the same number of lines"

        # Write JSONL with resized images
        count = 0
        with open(output_jsonl, 'w', encoding='utf-8') as fout:
            for _, row in df.iterrows():
                i = random.randrange(len(questions))
                selected_question = questions[i]
                txt_sum, txt_cap = summaries[i], captions[i]
                # metadata columns
                ignore_cols = {'Filename', 'Frame_No', 'Case_Summary', 'Detailed_Reasoning'}
                metadata_cols = [c for c in df.columns if c not in ignore_cols]
                meta_json = json.dumps(
                    {col: row[col] for col in metadata_cols},
                    ensure_ascii=False
                )
                formatted_answer = "\n".join([
                    txt_sum,
                    txt_cap,
                    f"<METADATA> {meta_json} </METADATA>",
                    f"<REASONING> {row['Detailed_Reasoning']} </REASONING>",
                    f"<CONCLUSION> {row['Case_Summary']} </CONCLUSION>"
                ])

                base = os.path.splitext(row['Filename'])[0]
                orig_path = os.path.join(image_dir, f"{base}_before.jpg")
                resized_path = os.path.join(resized_dir, f"{base}_before_512.jpg")

                if os.path.exists(orig_path):
                    with PILImage.open(orig_path) as img:
                        img = img.convert("RGB").resize((512, 512), PILImage.LANCZOS)
                        img.save(resized_path)
                else:
                    continue

                example = {
                    "system_message": instruction_str,
                    "messages": selected_question,
                    "answer": formatted_answer,
                    "image": resized_path
                }
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1

        # Load and push to HF Hub
        feature_dict = Features({
            'system_message': Value('string'),
            'messages': Value('string'),
            'answer': Value('string'),
            'image': Image()
        })
        ds = load_dataset('json', data_files=output_jsonl, split='train', features=feature_dict)
        ds.push_to_hub(repo_id=hf_repo_id, private=False)

        return f"✅ Wrote {count} examples to {output_jsonl} and pushed to {hf_repo_id}"

    except Exception as e:
        return f"❌ Error in simple dataset pipeline: {e}"


def create_dataset_maker_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Simple Orthodontic VLM Dataset Maker")
        gr.Markdown("This tool processes an Excel sheet and corresponding images to create a VLM-ready dataset in JSONL format and uploads it to the Hugging Face Hub.")
        with gr.Row():
            excel_input = gr.Textbox(label="Excel File Path", value="data/orthodontic_vlm.xlsx")
            image_dir_input = gr.Textbox(label="Original Image Directory", value="data/images/before")
            resized_dir_input = gr.Textbox(label="Resized Image Directory", value="data/images/before_512")
        with gr.Group():
            gr.Markdown("Text Resource Files")
            with gr.Row():
                instruction_input = gr.Textbox(label="Instruction Txt Path", value="data/text/instruction.txt")
                question_input = gr.Textbox(label="Question Txt Path", value="data/text/question.txt")
            with gr.Row():
                summary_input = gr.Textbox(label="Summary Txt Path", value="data/text/summary.txt")
                caption_input = gr.Textbox(label="Caption Txt Path", value="data/text/caption.txt")
        with gr.Group():
            gr.Markdown("Output Configuration")
            with gr.Row():
                output_jsonl_input = gr.Textbox(label="Output JSONL Path", value="output_vlm_dataset.jsonl")
                hf_repo_input = gr.Textbox(label="HF Repo ID", placeholder="username/my-vlm-dataset")
        
        run_button = gr.Button("Run Dataset Pipeline", variant="primary")
        output_box = gr.Textbox(label="Pipeline Output", interactive=False, lines=3)
        
        run_button.click(
            fn=run_simple_dataset,
            inputs=[
                excel_input, image_dir_input, resized_dir_input,
                instruction_input, question_input, summary_input, caption_input,
                output_jsonl_input, hf_repo_input
            ],
            outputs=[output_box]
        )
    return interface