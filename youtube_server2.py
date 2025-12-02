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
    # Film & Animation
    "A simple still-life photo of film reels and a clapperboard on a clean desk",
    "A minimal illustration-style drawing setup with sketch paper and pencils",
    
    # Autos & Vehicles
    "A close-up shot of a car dashboard with soft natural lighting",
    "A quiet scene of a parked bicycle near a wall in gentle daylight",
    
    # Music
    "A neatly arranged set of acoustic guitar strings and a tuning device on a wooden surface",
    "A pair of simple over-ear headphones placed on a clean white table",
    
    # Pets & Animals
    "A calm cat resting on a blanket beside a sunny window",
    "A friendly small dog sitting on a wooden floor in soft daylight",
    
    # Sports
    "A basketball resting on an empty court with warm afternoon light",
    "A pair of clean running shoes placed neatly on a wooden floor",
    
    # Travel & Events
    "A scenic landscape of a quiet lakeside with mountains in the distance",
    "A suitcase placed beside a softly lit window in a travel setting",
    
    # Gaming
    "A minimalist photo of a game controller on a wooden desk",
    "A clean setup with a gaming mouse and keyboard under soft light",
    
    # People & Blogs
    "A flat-lay photo of a notebook, a pen, and a cup of tea on a table",
    "A cozy scene with a book and a soft blanket near a window",
    
    # Entertainment
    "A simple arrangement of colorful stage lights casting soft beams on a wall",
    "A bowl of popcorn on a coffee table in front of a blank TV screen",
    
    # Howto & Style
    "A neatly organized set of makeup brushes on a plain white cloth",
    "A minimalist photo of folded towels and a small bottle of skincare lotion",
    
    # Education
    "A flat-lay of an open notebook with pencils and a clean wooden desk",
    "A stack of simple textbooks on a plain table in soft daylight",
    
    # Science & Technology
    "A clean workspace with a laptop, a notebook, and gentle natural light",
    "A close-up of electronic components arranged neatly on a desk"
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
