"""
Runner script for SOTA Manga Colorization using FLUX.2 Klein 4B + Reference LoRA.
"""
import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image
from diffusers import Flux2KleinPipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Paths
INPUT_PANEL = Path(r"C:\Users\ginto\.gemini\antigravity\brain\4145966c-5bb1-49af-990b-afcdfb8f3a2c\.user_uploaded\media_1791028028009.png")
REFERENCE_IMG = ROOT_DIR / "models" / "flux_manga_lora" / "nikaido_ref.png"
LORA_PATH = ROOT_DIR / "models" / "flux_manga_lora" / "manga_colorization.safetensors"
OUTPUT_PATH = Path(r"C:\Users\ginto\.gemini\antigravity\brain\4145966c-5bb1-49af-990b-afcdfb8f3a2c\flux2_sota_nikaido_result.png")

print("=== SOTA FLUX.2 KLEIN 4B MANGA COLORIZER ===")
print(f"Target Manga Panel: {INPUT_PANEL}")
print(f"Character Reference: {REFERENCE_IMG}")
print(f"LoRA Weights: {LORA_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print(f"Loading FLUX.2 Klein 4B on {device} ({dtype})...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=dtype
)
pipe.to(device)

print("Loading Manga Colorization LoRA adapter...")
pipe.load_lora_weights(str(LORA_PATH))

# Prepare Images
target_im = Image.open(INPUT_PANEL).convert("RGB")
ref_im = Image.open(REFERENCE_IMG).convert("RGB")

# Trigger word is mngclranm
prompt = "mngclranm, masterpiece, vibrant anime studio colors, natural lighting, clean lineart"

print("Running generation (4-step distilled rectified flow)...")
t0 = time.time()
result = pipe(
    image=[target_im, ref_im],
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

elapsed = time.time() - t0
print(f"Generated SOTA colorized panel in {elapsed:.2f} seconds!")

result.save(str(OUTPUT_PATH))
print(f"Saved result to: {OUTPUT_PATH}")
