from flask import Flask, request, Response, after_this_request
from flask_cors import CORS
from PIL import Image
import io
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from collections import deque
import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = "/workspace/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

result_queue = deque()
queue_lock = threading.Lock()
pipe_lock = threading.Lock()

blip_processor = None
blip_model = None
pipe = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

def init_models():
    global blip_processor, blip_model, pipe
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=CACHE_DIR)
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=dtype, cache_dir=CACHE_DIR).to(device).eval()
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=dtype, variant="fp16", use_safetensors=True, cache_dir=CACHE_DIR).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    with torch.no_grad():
        _ = pipe(prompt=["dummy"], num_inference_steps=1, guidance_scale=0.0).images

def build_prompt(caption):
    return f"A high-quality product image of {caption}, displayed on a plain white background with soft studio lighting. The item is centered and clearly visible, with no text, no watermark, and no packaging — just the product itself. Typical Amazon product listing style."

def generate_caption(image):
    with torch.no_grad():
        inputs = blip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=40)
        return blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def generate_images(index_caption_pairs):
    prompts = [build_prompt(caption) for _, caption in index_caption_pairs]
    with pipe_lock, torch.no_grad():
        images = pipe(prompt=prompts, num_inference_steps=1, guidance_scale=0.0).images
    results = []
    for (idx, _), img in zip(index_caption_pairs, images):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        results.append((idx, buffer.read()))
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
                print(f"[BLIP Error] ID={idx}, Error: {e}")
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

    response_parts = []
    for rect_id, jpeg_bytes in results_to_send:
        part = (
            f"--myboundary\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-ID: <{rect_id}>\r\n"
            f"\r\n"
        ).encode("utf-8") + jpeg_bytes + b"\r\n"
        response_parts.append(part)
    response_parts.append(b"--myboundary--\r\n")
    body = b"".join(response_parts)

    @after_this_request
    def clear_sent_queue(response):
        with queue_lock:
            for item in results_to_send:
                if item in result_queue:
                    result_queue.remove(item)
        return response

    return Response(body, status=200, headers={
        "Content-Type": "multipart/mixed; boundary=myboundary"
    })

if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=5000)
