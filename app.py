import os
import cv2
import gradio as gr
import numpy as np
from demo.inference import load_ensemble_models, predict_ensemble
from demo.mediapipe_extractor import MediaPipePoseExtractor

import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Loading Model on device: {DEVICE}...")
models = load_ensemble_models(device=DEVICE)
extractor = MediaPipePoseExtractor(model_complexity=0)
MANUAL_ROTATION = None
print(f"Loaded {list(models.keys())} models on {DEVICE}. Ready.")

def process_video_input(video_file):
    if video_file is None:
        return None, "Please upload a video file."

    skeletons, annotated_video_path = extractor.process_video_file(video_file, manual_rotation=MANUAL_ROTATION)

    if skeletons is None or len(skeletons) < 5:
        return None, "Error: Could not detect human pose landmarks in the uploaded video."

    # Run Ensemble Inference (Joint + Bone)
    predictions = predict_ensemble(models, skeletons, top_k=5, device=DEVICE)

    output_text = "### Hyper-GCN Ensemble Predictions:\n"
    for rank, (label, score) in enumerate(predictions, 1):
        output_text += f"**{rank}. {label.title()}**: `{score:.2f}%`  \n"

    return annotated_video_path, output_text

webcam_buffer = []
webcam_frame_count = 0
last_webcam_prediction = "Accumulating frames... (0/64 frames)"
webcam_history = []

WEBCAM_WARMUP_FRAMES = 30
WEBCAM_PREDICT_EVERY = 15
WEBCAM_HISTORY_KEEP = 8

def process_webcam_frame(frame_bgr):
    global webcam_buffer, webcam_frame_count, last_webcam_prediction, webcam_history

    if frame_bgr is None:
        return None, "Waiting for webcam feed..."

    joints, annotated_frame = extractor.extract_landmarks(frame_bgr)
    webcam_frame_count += 1

    if joints is not None:
        webcam_buffer.append(joints)
        if len(webcam_buffer) > 64:
            webcam_buffer.pop(0)

    ann_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    if len(webcam_buffer) < WEBCAM_WARMUP_FRAMES:
        output_text = (
            f"### Live Webcam\n"
            f"Accumulating motion... ({webcam_frame_count} frames, "
            f"{len(webcam_buffer)}/{WEBCAM_WARMUP_FRAMES} needed before prediction)\n"
        )
        last_webcam_prediction = output_text
    else:
        if webcam_frame_count % WEBCAM_PREDICT_EVERY == 0:
            seq_array = np.array(webcam_buffer, dtype=np.float32)
            predictions = predict_ensemble(models, seq_array, top_k=3, device=DEVICE)

            top_line = "  •  ".join(
                f"**{label.title()}** `{score:.1f}%`" for label, score in predictions[:3]
            )
            entry = f"Frame {webcam_frame_count}: {top_line}"
            webcam_history.append(entry)
            if len(webcam_history) > WEBCAM_HISTORY_KEEP:
                webcam_history.pop(0)

            current_text = "### Current Prediction (frame {}):\n".format(webcam_frame_count)
            for rank, (label, score) in enumerate(predictions, 1):
                current_text += f"**{rank}. {label.title()}**: `{score:.2f}%`  \n"

            history_text = "\n### Previous Results:\n" + "\n".join(
                f"- {h}" for h in webcam_history
            )
            last_webcam_prediction = current_text + history_text

        output_text = last_webcam_prediction

    return ann_frame_rgb, output_text

def clear_webcam_buffer():
    global webcam_buffer, webcam_frame_count, last_webcam_prediction, webcam_history
    webcam_buffer = []
    webcam_frame_count = 0
    webcam_history = []
    last_webcam_prediction = "Webcam buffer cleared."
    return last_webcam_prediction

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after {
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Hyper-GCN Action Recognition", theme=custom_theme, css=custom_css) as demo:
    gr.Markdown("# Hyper-GCN Action Recognition Demo")

    with gr.Tabs():
        with gr.TabItem("Video File Upload"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input Video Clip", sources=["upload"])
                    btn_run_video = gr.Button("Recognize Action", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Annotated Video Output")
                with gr.Column():
                    prediction_output_video = gr.Markdown(label="Top Predictions")

            btn_run_video.click(
                fn=process_video_input,
                inputs=[video_input],
                outputs=[video_output, prediction_output_video]
            )

        with gr.TabItem("Live Webcam Recognition"):
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam Feed")
                    btn_clear = gr.Button("Reset Webcam Buffer")
                with gr.Column():
                    webcam_output_image = gr.Image(label="Live Skeleton Detection")
                with gr.Column():
                    prediction_output_webcam = gr.Markdown(label="Live Predictions")

            webcam_input.stream(
                fn=process_webcam_frame,
                inputs=[webcam_input],
                outputs=[webcam_output_image, prediction_output_webcam]
            )
            btn_clear.click(fn=clear_webcam_buffer, inputs=[], outputs=[prediction_output_webcam])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=1234)