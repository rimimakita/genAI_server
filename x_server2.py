from flask import Flask, request, Response, after_this_request
from flask_cors import CORS
from PIL import Image
import io
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from collections import deque
import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import random

CACHE_DIR = "/workspace/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

result_queue = deque()
queue_lock = threading.Lock()
pipe_lock = threading.Lock()

pipe = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

def init_models():
    global pipe
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        cache_dir=CACHE_DIR
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        _ = pipe(prompt=["dummy"], height=448, width=448, num_inference_steps=1, guidance_scale=0.0).images

# ランダム語彙
prompt = [
    # Technology
    "A clean minimalist desk with a laptop and a coffee cup under soft natural light",
    "A modern workspace with a keyboard, wireless mouse, and a small potted plant",
    "A close-up of a sleek smartphone on a neutral-colored desk",

    # Animals / Pets
    "A small cat resting quietly on a blanket near a sunny window",
    "A friendly dog sitting on a wooden floor illuminated by soft daylight",
    "A peaceful bird perched on a simple branch against a blurred green background",

    # Food
    "A neatly arranged plate of fresh fruit on a wooden table in natural morning light",
    "A cup of coffee and a small pastry placed on a white dish in a top-down view",
    "A clean minimal plate with a small slice of cake arranged neatly on a wooden table in natural morning light",

    # Travel
    "A serene lakeside landscape with mountains in the distance",
    "A simple travel scene with a suitcase placed beside a softly lit window",
    "A beach shoreline with gentle waves under a clear sky",
    # Added
    "A city skyline viewed from across a river",
    "A wide landscape with distant hills and open fields",
    "A coastal walkway with the ocean stretching into the horizon",

    # Urban / Minimal City
    "A quiet street corner with soft morning light and minimal traffic",
    "A minimal urban skyline with muted colors under a clear sky",
    # Added
    "A city street with buildings and pedestrians in the distance",
    "A crosswalk scene with cars and people going about their day",
    "A row of shops along a busy sidewalk",
    "A view of an apartment building against a blue sky",
    "A small café on a street corner with outdoor seating",
    "A residential street lined with houses and parked cars",

    # Indoor / Lifestyle
    "A tidy room with sunlight coming through the window",
    "A shelf with books, plants, and small decorations",
    "A wooden table beside a window with light coming in",

    # Art / Photography
    "A minimalist still-life arrangement of dried flowers on a plain table",
    "Soft natural light casting gentle shadows across a simple interior space",
    "A flat-lay photo of a sketchbook and pencils arranged neatly on a wooden desk",

    # Added Abstract / Mood
    "Colorful lights shining in a dark space",
    "A gradient sky transitioning from blue to orange behind buildings"
]


def generate_random_caption(seed=None):
    rnd = random.Random(seed) if seed is not None else random
    return rnd.choice(prompt)
# def generate_random_caption(seed=None): 
#     rnd = random.Random(seed) if seed is not None else random 
#     return f"{rnd.choice(ADJECTIVES)} {rnd.choice(NOUNS)}"

def build_prompt(caption: str) -> str:
    # ★ caption をそのまま使う
    return caption


def generate_images(index_caption_pairs):
    prompts = [build_prompt(caption) for _, caption in index_caption_pairs]
    with pipe_lock, torch.no_grad():
        images = pipe(prompt=prompts, height=512, width=512, num_inference_steps=1, guidance_scale=0.0).images
    results = []
    for (idx, caption), img in zip(index_caption_pairs, images):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        results.append((idx, buffer.read(), caption))
    with queue_lock:
        for item in results:
            result_queue.append(item)

def split_into_batches(indexed_list, batch_size=3):
    return [indexed_list[i:i+batch_size] for i in range(0, len(indexed_list), batch_size)]

def async_process_and_store(decoded_images):
    # BLIP-2 を使わず、単にランダム語句を割り当てる
    results = []
    for idx, _img in decoded_images:
        # 再現性が欲しければ seed=hash(idx) にする
        caption = generate_random_caption()
        results.append((idx, caption))
    for batch in split_into_batches(results):
        generate_images(batch)

@app.route("/process_batch", methods=["POST"])
def process_batch():
    files = request.files
    id_image_pairs = [(rect_id, files[rect_id].read()) for rect_id in files]

    def worker(pairs):
        decoded = []
        for rect_id, raw in pairs:
            try:
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                decoded.append((rect_id, image))
            except Exception as e:
                print(f"[Decode Error] ID={rect_id}, Error: {e}")
        async_process_and_store(decoded)

    threading.Thread(target=worker, args=(id_image_pairs,)).start()
    return "Accepted", 202

@app.route("/get_results", methods=["GET"])
def get_all_results():
    timeout = 5.0
    poll_interval = 0.05
    waited = 0
    results_to_send = []

    while waited < timeout:
        with queue_lock:
            if result_queue:
                results_to_send = list(result_queue)
                break
        time.sleep(poll_interval)
        waited += poll_interval

    if not results_to_send:
        return "", 204

    json_data = []
    for rect_id, jpeg_bytes, caption in results_to_send:
        b64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
        json_data.append({
            "id": rect_id,
            "image": b64_image,
            "caption": caption
        })

    @after_this_request
    def clear_sent_queue(response):
        with queue_lock:
            for item in results_to_send:
                if item in result_queue:
                    result_queue.remove(item)
        return response

    return Response(
        json.dumps(json_data),
        status=200,
        mimetype="application/json"
    )

if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=5000)
