from flask import Flask, request, Response, after_this_request
from flask_cors import CORS
from PIL import Image
import io
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from collections import deque
import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json

CACHE_DIR = "/workspace/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

result_queue = deque()
queue_lock = threading.Lock()
pipe_lock = threading.Lock()

caption_processor = None
caption_model = None
pipe = None
keywords = ["text", "numbers", "logo","wall"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

def init_models():
    global caption_processor, caption_model, pipe
    caption_processor = AutoProcessor.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        cache_dir=CACHE_DIR,
        use_fast=False  # ← ここを追加！
    )
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=CACHE_DIR
    ).eval()
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

    prompt_caption = caption
    for kw in keywords:
        prompt_caption = prompt_caption.replace(kw, "object").replace(kw.capitalize(), "object")
    return f"A photo of a {prompt_caption} item on a white background, centered, no text, no shadow, no packaging."

def generate_caption(image):
    with torch.no_grad():
        inputs = caption_processor(images=image, return_tensors="pt").to(device)
        generated_ids = caption_model.generate(**inputs, max_new_tokens=30)
        caption = caption_processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()

def generate_images(index_caption_pairs):
    prompts = [build_prompt(caption) for _, caption in index_caption_pairs]
    with pipe_lock, torch.no_grad():
        images = pipe(prompt=prompts, height=512, width=512, num_inference_steps=1, guidance_scale=0.0).images
    results = []
    for (idx, caption), img in zip(index_caption_pairs, images):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        results.append((idx, buffer.read(), caption))  # ← キャプションも追加
    with queue_lock:
        for item in results:
            result_queue.append(item)

def split_into_batches(indexed_list, batch_size=3):
    return [indexed_list[i:i+batch_size] for i in range(0, len(indexed_list), batch_size)]

def async_process_and_store(decoded_images):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_caption, img): idx for idx, img in decoded_images}
        results = []
        for future in as_completed(futures):
            idx = futures[future]
            try:
                caption = future.result()
                results.append((idx, caption))
            except Exception as e:
                print(f"[Caption Error] ID={idx}, Error: {e}")
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

    # JSONデータを作成
    json_data = []
    for rect_id, jpeg_bytes, caption in results_to_send:
        b64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
        json_data.append({
            "id": rect_id,
            "image": b64_image,
            "caption": caption
        })

    # 送信済みキューから削除
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
