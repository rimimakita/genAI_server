from flask import Flask, request, Response
from flask_cors import CORS
from PIL import Image
import io
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import AutoPipelineForText2Image
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading
import os  # ← これを追加

# ✅ ここにキャッシュディレクトリの指定を追加
CACHE_DIR = "/workspace/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# グローバル状態
result_queue = deque()
queue_lock = threading.Lock()

# モデル関連を初期は None に
blip_processor = None
blip_model = None
pipe = None

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

def init_models():
    global blip_processor, blip_model, pipe
    blip_processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir=CACHE_DIR
    )
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=dtype,
        cache_dir=CACHE_DIR
    ).to(device).eval()
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", 
        torch_dtype=dtype, 
        variant="fp16", 
        use_safetensors=True,
        cache_dir=CACHE_DIR
    ).to(device)

    # 任意：ウォームアップ（必要な場合のみ）
    with torch.no_grad():
        _ = pipe(prompt=["dummy"], num_inference_steps=1, guidance_scale=0.0).images

def build_prompt(caption):
    return f"A high-quality product image of {caption}, displayed on a plain white background with soft studio lighting. The item is centered and clearly visible, with no text, no watermark, and no packaging — just the product itself. Typical Amazon product listing style."

def process_image_batch(indexed_images):
    indices, images = zip(*indexed_images)
    with torch.no_grad():
        inputs = blip_processor(images=list(images), return_tensors="pt").to(device)
        generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=40)
        captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)

    prompts = [build_prompt(c.strip()) for c in captions]
    with torch.no_grad():
        generated_images = pipe(prompt=prompts, num_inference_steps=1, guidance_scale=0.0).images

    results = []
    for idx, img in zip(indices, generated_images):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        results.append((idx, buffer.read()))

    # メモリ開放（必要なら）
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    return results

def split_into_batches(indexed_images, max_batch_size=3):
    return [indexed_images[i:i+max_batch_size] for i in range(0, len(indexed_images), max_batch_size)]

def async_process_and_store(id_image_pairs):
    batches = split_into_batches(id_image_pairs, max_batch_size=3)
    with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
        futures = executor.map(process_image_batch, batches)

    results = []
    for batch_result in futures:
        results.extend(batch_result)

    with queue_lock:
        for rect_id, jpeg_bytes in results:
            print(f"[Server] Add to queue: ID={rect_id}, size={len(jpeg_bytes)} bytes")
            result_queue.append((rect_id, jpeg_bytes))

@app.route("/process_batch", methods=["POST"])
def process_batch():
    files = request.files
    id_image_pairs = []
    for rect_id in files:
        raw = files[rect_id].read()  # バイナリのまま受け取る（変換は後回し）
        id_image_pairs.append((rect_id, raw))

    def worker(pairs):
        decoded = []
        for rect_id, raw in pairs:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            decoded.append((rect_id, image))
        async_process_and_store(decoded)

    threading.Thread(target=worker, args=(id_image_pairs,)).start()
    return "Accepted", 202

@app.route("/get_results", methods=["GET"])
def get_all_results():
    with queue_lock:
        if not result_queue:
            print("[Server] Queue is empty")
            return "", 204
        results_to_send = list(result_queue)
        result_queue.clear()

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

    return Response(body, status=200, headers={
        "Content-Type": "multipart/mixed; boundary=myboundary"
    })

if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=5000)
