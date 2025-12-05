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
    "A movie screen glowing in a dark room",
    "A desk with storyboard papers and film equipment",
    "Lights and shadows in a small filming studio",

    # Autos & Vehicles
    "A car parked on a city street in the morning",
    "A bicycle beside a wall on a sidewalk",
    "A road lined with buildings and trees",

    # Music
    "Colorful stage lights in front of a dark background",
    "A speaker playing music in a room",
    "A corner with a guitar and music sheets",

    # Pets & Animals
    "A cat lying on a bed near a window",
    "A dog sitting in a bright room",
    "A bird perched on a branch",

    # Sports
    "A basketball on an outdoor court",
    "Running shoes near a window",
    "An empty sports field",

    # Travel & Events
    "A lake with mountains in the background",
    "A hotel room with a window showing a city view",
    "A beach with waves rolling onto the sand",

    # Gaming
    "A gaming desk with a monitor and controller",
    "A computer setup with LED lights",
    "A gaming corner with a keyboard and mouse",

    # People & Blogs
    "A morning scene inside a bright room",
    "A desk with a notebook and a cup of tea",
    "A living room with books and small decorations",

    # Entertainment
    "Colorful lights shining in a dark room",
    "A living room with a TV turned on",
    "A bright pattern of lights on a stage",

    # Howto & Style
    "A table with makeup items arranged neatly",
    "A workspace with tools on a table",
    "A room with clothing and accessories on a chair",

    # Education
    "A study desk with books and stationery",
    "A table with notebooks and pencils",
    "A chalkboard with writing",

    # Science & Technology
    "A desk with a laptop and cables",
    "Electronic parts arranged on a table",
    "A modern workspace with a computer",

    # Food / Café
    "A slice of cake on a plate in a café",
    "A cup of coffee on a table by a window",
    "Fruit and bread arranged on a breakfast table",
    "The inside of a café with tables and chairs",
    "A table with fresh fruit and a drink",

    # Sky + City（空だけを避ける）
    "A blue sky above city buildings",
    "Clouds over a row of houses",
    "A sunset behind an urban skyline",
    "A sky with scattered clouds above a shopping street",

    # Night City
    "A city street at night with bright signs",
    "Buildings lit up at night",
    "Neon lights glowing in a busy area",
    "A quiet alley with streetlights at night",

    # Urban / Street Scenes
    "A residential street with houses and trees",
    "A town square with shops and people walking",
    "A café street with outdoor seating",
    "A pedestrian walkway between buildings",
    "A corner of a city street with storefronts",
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

