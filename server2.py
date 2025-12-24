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
ADJECTIVES = [
"A clear plastic bottle of mineral water",
"A white disposable nonwoven face mask",
"A large white plastic container of liquid detergent",
"A paperback book with a colorful cover design",
"A stack of soft cotton towels",
"A compact handheld hair removal device",
"A pair of soft indoor slippers made of foam material",
"A pair of thick ankle socks for cold weather",
"A slim cosmetic eyeliner pencil with a cap",
"A refill pouch of liquid hand soap with printed label",
"A small blue USB flash drive",
"A black USB-C cable with connectors on both ends",
"A resealable pouch of whey protein powder",
"A refill pack of liquid laundry detergent",
"A plastic case of a video game for a handheld console",
"A rectangular prepaid gift card with printed value"
]


amazon_prompts = [
    # Amazon Devices & Accessories
    "a protective case for a tablet",
    "a protective film for a smartphone",

    # DIY, Tools & Garden
    "a metal hammer",
    "a gardening trowel",

    # PC Software / PC-related objects
    "a USB flash drive",

    # Computers & Peripherals (expanded)
    "a laptop with a closed lid",
    "a desktop monitor",
    "a wireless keyboard",
    "a wireless mouse",

    # Apps & Games / Gaming devices
    "a game controller",
    "a portable game console",
    "a home video game console",

    # Toys
    "wooden building blocks",

    # Gift Cards
    "a plastic gift card",

    # Sports & Outdoors
    "a collapsible water bottle",

    # Digital Music
    "over-ear headphones",

    # Drugstore（★シャンプー系 半減）
    "a shampoo bottle",
    "a conditioner bottle",
    "a hair care bottle",
    "a white container of skincare cream",

    # Beauty
    "a makeup brush",
    "a compact mirror",

    # Fashion
    "a fabric tote bag",
    "cloth slippers",
    "a short-sleeve T-shirt",
    "folded towels",

    # Home & Kitchen
    "a ceramic mug",
    "a wooden cutting board",

    # Hobby
    "drawing pencils",
    "a brush for model building",

    # Music
    "acoustic guitar strings",

    # Large Appliances
    "an electric kettle",
    "a table fan",

    # Electronics & Cameras
    "a digital camera",
    "a desk lamp",

    # Office Supplies
    "a notebook and a ballpoint pen",
    "metal paper clips",

    # Books
    "a book with a plain cover",

    # Musical Instruments
    "an electronic metronome",
    "a recorder flute",

    # Automotive
    "a smartphone holder for cars",
    "a helmet bag for motorcycles",

    # Food, Drinks & Alcohol（★少し増やした）
    "a bottle of mineral water",
    "a packaged chocolate bar",
    "a pack of crackers",
    "a bag of roasted nuts"
]



def generate_random_caption(seed=None):
    rnd = random.Random(seed) if seed is not None else random
    return rnd.choice(amazon_prompts)
# def generate_random_caption(seed=None): 
#     rnd = random.Random(seed) if seed is not None else random 
#     return f"{rnd.choice(ADJECTIVES)} {rnd.choice(NOUNS)}"

def build_prompt(caption):
    return f"A photo of a {caption} item on a white background, centered, no text, no shadow, no packaging."


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
