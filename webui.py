import os
import sys
import time
import subprocess
import cv2
import numpy as np
from PIL import Image
import gradio as gr

CLI_EXE = r"tools\sd_cpp\sd-cli.exe"
DIFF_MODEL = "models/flux_gguf/flux1-schnell-Q4_0.gguf"
CLIP_L = "models/flux_gguf/clip_l.safetensors"
T5XXL = "models/flux_gguf/t5-v1_1-xxl-encoder-Q3_K_M.gguf"
VAE = "models/flux_gguf/ae.safetensors"
LORA_DIR = "models/flux_manga_lora"

os.makedirs("webui_scratch", exist_ok=True)

def generate_txt2img(prompt, neg_prompt, width, height, steps, cfg_scale, lora_weight, seed):
    out_path = f"webui_scratch/txt2img_{int(time.time()*1000)}.png"
    
    full_prompt = prompt
    if lora_weight > 0:
        full_prompt += f" <lora:anime_schnell:{lora_weight}>"
        
    cmd = [
        CLI_EXE,
        "--diffusion-model", DIFF_MODEL,
        "--clip_l", CLIP_L,
        "--t5xxl", T5XXL,
        "--vae", VAE,
        "--lora-model-dir", LORA_DIR,
        "--prompt", full_prompt,
        "-W", str(int(width)),
        "-H", str(int(height)),
        "--steps", str(int(steps)),
        "--cfg-scale", str(float(cfg_scale)),
        "-s", str(int(seed) if seed >= 0 else -1),
        "-o", out_path
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - t0
    
    if os.path.exists(out_path):
        img = Image.open(out_path)
        info = f"Generated in {elapsed:.1f}s | Resolution: {int(width)}x{int(height)} | Steps: {steps} | Model: FLUX.1 Schnell GGUF (Vulkan GPU)"
        return img, info
    return None, "Error during generation"

def generate_img2img(input_image, prompt, neg_prompt, strength, width, height, steps, cfg_scale, lora_weight, preserve_lines, clean_bubbles, seed):
    if input_image is None:
        return None, None, "Please upload an image first."
        
    t_id = int(time.time()*1000)
    init_path = f"webui_scratch/init_{t_id}.png"
    out_path = f"webui_scratch/out_{t_id}.png"
    
    # Save input
    orig_np = np.array(input_image)
    if len(orig_np.shape) == 2:
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        
    h_orig, w_orig = orig_bgr.shape[:2]
    
    # Target resolution (multiples of 16/32)
    target_w = int(width)
    target_h = int(height)
    
    pil_in = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    pil_in.resize((target_w, target_h), Image.Resampling.LANCZOS).save(init_path)
    
    full_prompt = prompt
    if lora_weight > 0:
        full_prompt += f" <lora:anime_schnell:{lora_weight}>"
        
    cmd = [
        CLI_EXE,
        "--diffusion-model", DIFF_MODEL,
        "--clip_l", CLIP_L,
        "--t5xxl", T5XXL,
        "--vae", VAE,
        "--lora-model-dir", LORA_DIR,
        "--init-img", init_path,
        "--strength", str(float(strength)),
        "--prompt", full_prompt,
        "-W", str(target_w),
        "-H", str(target_h),
        "--steps", str(int(steps)),
        "--cfg-scale", str(float(cfg_scale)),
        "-s", str(int(seed) if seed >= 0 else -1),
        "-o", out_path
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.time() - t0
    
    if not os.path.exists(out_path):
        return None, None, "Error executing FLUX img2img"
        
    flux_raw = cv2.imread(out_path)
    flux_fit = cv2.resize(flux_raw, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
    
    if preserve_lines:
        gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        color_f = flux_fit.astype(np.float32) / 255.0
        raw_f = orig_bgr.astype(np.float32) / 255.0
        mult_bgr = (color_f * raw_f * 255.0).astype(np.uint8)
        
        ink_mask = (gray < 85).astype(np.float32)[:, :, None]
        final_bgr = (mult_bgr.astype(np.float32) * (1.0 - ink_mask) + orig_bgr.astype(np.float32) * ink_mask).astype(np.uint8)
        
        if clean_bubbles:
            white_bg = (gray > 225).astype(np.uint8) * 255
            ink_bin = (gray < 85).astype(np.uint8) * 255
            white_bg[ink_bin > 0] = 0
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_bg)
            for i in range(1, num_labels):
                x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                if (x <= 5 or y <= 5 or x + bw >= w_orig - 5 or y + bh >= h_orig - 5 or area > 1000):
                    if float(np.mean(gray[labels == i])) > 230:
                        final_bgr[labels == i] = orig_bgr[labels == i]
    else:
        final_bgr = flux_fit
        
    final_rgb = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
    flux_raw_rgb = Image.fromarray(cv2.cvtColor(flux_fit, cv2.COLOR_BGR2RGB))
    
    info = f"Generated in {elapsed:.1f}s | Denoising Strength: {strength} | Steps: {steps} | FLUX Schnell + Anime LoRA"
    return final_rgb, flux_raw_rgb, info

# A1111 Classic Dark Web Theme CSS
custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.header-bar {
    background: #1e293b;
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 12px;
    border: 1px solid #334155;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.header-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f8fafc;
}
.header-badge {
    background: #0284c7;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.generate-btn {
    background: #ea580c !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 8px !important;
    height: 52px !important;
}
.generate-btn:hover {
    background: #c2410c !important;
}
.panel-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 12px;
}
"""

with gr.Blocks(title="Stable Diffusion WebUI - FLUX.1 + LoRA") as demo:
    gr.HTML("""
    <div class="header-bar">
        <div>
            <span class="header-title">Stable Diffusion WebUI (AUTOMATIC1111 Style)</span>
            <span style="margin-left: 10px; color: #94a3b8; font-size: 0.9rem;">Model: <b>FLUX.1-schnell-Q4_0.gguf</b></span>
        </div>
        <div>
            <span class="header-badge">AMD Radeon RX 9060 XT (Vulkan)</span>
            <span class="header-badge" style="background: #16a34a; margin-left: 5px;">Active LoRA: anime_schnell</span>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: img2img (Manga Colorization & Style Transfer)
        with gr.TabItem("img2img (Manga Colorizer)"):
            with gr.Row():
                with gr.Column(scale=5):
                    prompt_i2i = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        value="clean anime manga colorization, dorohedoro nikaido blonde hair, fair smooth skin, blush, toned cleavage, exact anime colors",
                        placeholder="Enter prompt here..."
                    )
                    neg_prompt_i2i = gr.Textbox(
                        label="Negative prompt",
                        lines=2,
                        value="blurry, distorted lineart, ugly, bad anatomy, deformed eyes, extra limbs, noise",
                        placeholder="Enter negative prompt..."
                    )
                    
                    init_img_input = gr.Image(label="Source Image (Manga Page / Panel)", type="pil")
                    
                    with gr.Accordion("Generation Parameters", open=True):
                        with gr.Row():
                            steps_i2i = gr.Slider(minimum=1, maximum=12, value=4, step=1, label="Sampling Steps (Schnell default: 4)")
                            cfg_i2i = gr.Slider(minimum=1.0, maximum=7.0, value=1.0, step=0.5, label="CFG Scale (FLUX: 1.0)")
                        with gr.Row():
                            width_i2i = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Target Width")
                            height_i2i = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Target Height")
                        with gr.Row():
                            strength_i2i = gr.Slider(minimum=0.05, maximum=0.85, value=0.15, step=0.05, label="Denoising Strength (0.15 for exact lineart, 0.4+ for redraw)")
                            lora_w_i2i = gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.1, label="Anime LoRA Weight")
                        with gr.Row():
                            preserve_ink = gr.Checkbox(value=True, label="Preserve 100% Original Ink Lines (Optical Multiply)")
                            clean_bubbles = gr.Checkbox(value=True, label="Clean Speech Bubbles & Gutters to Pure White")
                        seed_i2i = gr.Number(value=-1, label="Seed (-1 for Random)", precision=0)
                        
                    btn_i2i = gr.Button("Generate", elem_classes=["generate-btn"])
                    
                with gr.Column(scale=5):
                    out_final_i2i = gr.Image(label="Final Colored Result (Composited)", type="pil")
                    with gr.Accordion("Raw FLUX Diffusion Output (Before Ink Composite)", open=False):
                        out_raw_i2i = gr.Image(label="Raw FLUX Output", type="pil")
                    info_i2i = gr.Textbox(label="Generation Details", interactive=False)

            btn_i2i.click(
                fn=generate_img2img,
                inputs=[init_img_input, prompt_i2i, neg_prompt_i2i, strength_i2i, width_i2i, height_i2i, steps_i2i, cfg_i2i, lora_w_i2i, preserve_ink, clean_bubbles, seed_i2i],
                outputs=[out_final_i2i, out_raw_i2i, info_i2i]
            )
            
        # TAB 2: txt2img (Text to Image)
        with gr.TabItem("txt2img"):
            with gr.Row():
                with gr.Column(scale=5):
                    prompt_t2i = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        value="masterpiece anime character, nikaido with blonde hair, fair smooth skin, anime studio key visual",
                        placeholder="Enter prompt here..."
                    )
                    neg_prompt_t2i = gr.Textbox(
                        label="Negative prompt",
                        lines=2,
                        value="lowres, bad anatomy, bad hands, cropped, worst quality",
                        placeholder="Enter negative prompt..."
                    )
                    with gr.Row():
                        steps_t2i = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Sampling Steps")
                        cfg_t2i = gr.Slider(minimum=1.0, maximum=5.0, value=1.0, step=0.5, label="CFG Scale")
                    with gr.Row():
                        width_t2i = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                        height_t2i = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                    with gr.Row():
                        lora_w_t2i = gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.1, label="Anime LoRA Weight")
                        seed_t2i = gr.Number(value=-1, label="Seed (-1 for Random)", precision=0)
                    btn_t2i = gr.Button("Generate", elem_classes=["generate-btn"])
                with gr.Column(scale=5):
                    out_t2i = gr.Image(label="Output Image", type="pil")
                    info_t2i = gr.Textbox(label="Generation Details", interactive=False)
                    
            btn_t2i.click(
                fn=generate_txt2img,
                inputs=[prompt_t2i, neg_prompt_t2i, width_t2i, height_t2i, steps_t2i, cfg_t2i, lora_w_t2i, seed_t2i],
                outputs=[out_t2i, info_t2i]
            )

if __name__ == "__main__":
    print("Starting Stable Diffusion WebUI on http://127.0.0.1:7860...")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Default(primary_hue="orange", neutral_hue="slate"), css=custom_css)
