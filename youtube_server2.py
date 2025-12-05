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
from concurrent.futures import ThreadPoolExecutor, as_completed  # 使っていなければ削除してOK
import base64
import json
import random

# =====================
# 基本セットアップ
# =====================
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
        cache_dir=CACHE_DIR,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # ウォームアップ
    with torch.no_grad():
        _ = pipe(
            prompt=["dummy"],
            height=448,
            width=448,
            num_inference_steps=1,
            guidance_scale=0.0,
        ).images


# =====================
# ランダム語彙（プロンプト候補）
# =====================
prompt = [
    # Film & Animation
    "Soft cinematic lighting in a dim room with gentle dust particles floating in the air",
    "A simple movie-watching scene with a softly glowing screen lighting up a cozy dark room",
    "A warm-toned filmmaking workspace with scattered storyboard papers under natural light",

    # Autos & Vehicles
    "A quiet street at sunrise with a single parked car and long soft shadows",
    "A minimal roadside scene with a bicycle leaning against a textured wall in gentle daylight",
    "A peaceful suburban road lined with trees under soft morning light",

    # Music
    "Soft colorful stage lights creating a calm ambient atmosphere",
    "A cozy room with a speaker playing soft music and warm light entering from a window",
    "A simple music practice corner with faint light and minimal instruments",

    # Pets & Animals
    "A cat resting in a sunlit room with warm soft tones",
    "A dog sitting calmly near a bright window with natural morning light",
    "A quiet scene of a bird perched on a branch against a blurred background",

    # Sports
    "A quiet empty sports field with soft afternoon sunlight",
    "A pair of running shoes near a sunlit window in a minimal room",
    "A basketball court with warm sunset light and long shadows",

    # Travel & Events
    "A peaceful lakeside landscape with mountains under clear soft light",
    "A sunlit hotel room with an open window showing a distant city view",
    "A calm beach scene with gentle waves and a pale sky",

    # Gaming
    "A dimly lit gaming setup with ambient LED lighting in a cozy room",
    "A minimalist desk with a controller beside a softly glowing monitor",
    "A warm, inviting gaming corner with subtle colored lights",

    # People & Blogs
    "A cozy morning scene with soft window light illuminating a tidy room",
    "A warm desk corner with tea, notebooks, and gentle daylight",
    "A relaxed living room atmosphere with natural light and minimalist decor",

    # Entertainment
    "Soft colorful ambient lights casting gentle beams in a dark room",
    "A simple movie night scene with warm screen light reflecting in a cozy living room",
    "A fun, colorful background pattern with blurred vibrant lights",

    # Howto & Style
    "A clean vanity table with soft daylight and organized personal items",
    "A minimal workspace with neatly arranged tools under natural light",
    "A cozy room scene with fabrics and accessories in warm sunlight",

    # Education
    "A quiet study desk with open books and natural afternoon light",
    "A clean workspace with stationery and gentle soft shadows",
    "A simple chalkboard scene with warm ambient lighting",

    # Science & Technology
    "A modern workspace with a laptop in a softly lit minimal room",
    "A close-up of electronic parts on a clean surface with soft light",
    "A futuristic but minimal tech desk setup in natural lighting",

    # --- Added: Food / Café ---
    "A neatly arranged slice of cake on a white plate in soft natural light",
    "A cup of coffee on a wooden table with warm morning sunlight",
    "A simple breakfast scene with toast and fruit in a bright cozy kitchen",
    "A minimalist café interior with a small plant and sunlight through the window",
    "A bowl of fresh fruit arranged simply on a wooden surface in gentle daylight",

    # --- Added: Sky / Weather ---
    "A calm sky with soft pastel clouds during early morning light",
    "A gentle sunset with warm colors fading into a clear horizon",
    "A bright blue sky with soft white clouds moving slowly",
    "A dramatic cloudy sky with sunlight breaking through",

    # --- Added: Night City ---
    "A quiet city street at night with warm glowing streetlights",
    "A minimal night skyline with soft reflections on wet pavement",
    "A cozy alleyway illuminated by dim lanterns and warm tones",
    "A distant cityscape with colorful neon lights in the evening",

    # --- Added: Urban / Street Scenes ---
    "A calm residential street with trees and warm afternoon light",
    "A peaceful town square with soft sunlight and minimal activity",
    "A café street with outdoor seating in gentle daylight",
    "A quiet pedestrian walkway lined with buildings in soft morning light",
    "A simple urban corner with shadows cast by tall buildings",
]




def generate_random_caption(seed=None):
    rnd = random.Random(seed) if seed is not None else random
    return rnd.choice(prompt)


def build_prompt(caption: str) -> str:
    # caption をそのまま使う
    return caption


def generate_images(index_caption_pairs):
    prompts = [build_prompt(caption) for _, caption in index_caption_pairs]
    with pipe_lock, torch.no_grad():
        images = pipe(
            prompt=prompts,
            height=512,
            width=512,
            num_inference_steps=1,
            guidance_scale=0.0,
        ).images

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
    return [indexed_list[i:i + batch_size] for i in range(0, len(indexed_list), batch_size)]


def async_process_and_store(decoded_images):
    # BLIP-2 を使わず、単にランダム語彙を割り当てる
    results = []
    for idx, _img in decoded_images:
        caption = generate_random_caption()
        results.append((idx, caption))

    for batch in split_into_batches(results):
        generate_images(batch)


# =====================
# Flask routes
# =====================
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
    waited = 0.0
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
        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
        json_data.append(
            {
                "id": rect_id,
                "image": b64_image,
                "caption": caption,
            }
        )

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
        mimetype="application/json",
    )


# =====================
# エントリーポイント
# =====================
if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=5000)

