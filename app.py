import os
import io
import base64
import time
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, Response
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "WHITE" / "resized"
OUTPUT_BASE = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Per-image prompts: {target_objects} describes what to apply the fabric to
IMAGE_PROMPTS = {
    "WHITE 3.jpg": {
        "target": "the entire fabric and button closure area (but not the button itself)",
        "desc": "Fabric & Button",
    },
    "WHITE 4.jpg": {
        "target": "only the top-left fabric piece and the fabric tail/loop",
        "desc": "Fabric Corner & Tail",
    },
    "WHITE 5.jpg": {
        "target": "only the pillow (the pillowcase fabric)",
        "desc": "Pillow Close-up",
    },
    "WHITE 6.jpg": {
        "target": "the entire bed linen including the duvet cover, bed sheet, and all pillows",
        "desc": "Full Bed Set",
    },
    "WHITE 7.jpg": {
        "target": "the top part of the bed - the fitted sheet fabric on top of the mattress",
        "desc": "Fitted Sheet Top",
    },
    "WHITE 9.jpg": {
        "target": "the bottom-left side elastic/gathered fabric",
        "desc": "Elastic Fabric",
    },
    "WHITE 10 ABC.jpg": {
        "target": "the bolster (the cylindrical pillow)",
        "desc": "Bolster",
    },
    "WHITE 11 APC.jpg": {
        "target": "both pillows",
        "desc": "Pillows",
    },
    "WHITE FIT 1.jpg": {
        "target": "the entire bed sheet, all pillows, the bolster, and the blanket",
        "desc": "Fitted Bed Set",
    },
    "WHITE QLT 1.jpg": {
        "target": "the bed duvet/quilt, all pillows, and the blanket",
        "desc": "Quilt Bed Set",
    },
    "WHITE QLT 2.jpg": {
        "target": "the entire center bed including the duvet/quilt, all pillows, and blankets",
        "desc": "Quilt Center Bed",
    },
}


def build_prompt(target: str) -> str:
    """Build the Gemini prompt for applying a fabric sample to target objects."""
    return (
        f"I'm providing two images. The FIRST image is a fabric sample showing the color/pattern I want. "
        f"The SECOND image is a product photo with white bedding.\n\n"
        f"Apply the fabric sample's color and pattern to {target} in the product photo. "
        f"The fabric should wrap realistically following the folds, creases and contours of the original white fabric. "
        f"Preserve all shadows, highlights, and depth. "
        f"Keep everything else in the scene (background, furniture, walls, floor, decorations, non-target items) exactly unchanged. "
        f"The result must look like a natural, professional product photo - not a flat overlay."
    )


def recolor_image(client, sample_bytes: bytes, sample_mime: str, image_path: Path, prompt: str) -> bytes | None:
    """Send fabric sample + product image to Gemini. Returns image bytes or None."""
    with open(image_path, "rb") as f:
        product_bytes = f.read()

    product_mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[
            types.Part.from_bytes(data=sample_bytes, mime_type=sample_mime),
            types.Part.from_bytes(data=product_bytes, mime_type=product_mime),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    return None


@app.route("/")
def index():
    return FRONTEND_HTML


@app.route("/api/images")
def list_images():
    """Return the list of template images with descriptions."""
    result = []
    for filename in sorted(IMAGE_PROMPTS.keys()):
        result.append({
            "filename": filename,
            "desc": IMAGE_PROMPTS[filename]["desc"],
        })
    return jsonify(result)


@app.route("/api/preview/<path:filename>")
def preview_image(filename):
    """Serve original template images for preview."""
    return send_from_directory(str(INPUT_DIR), filename)


@app.route("/api/upload-sample", methods=["POST"])
def upload_sample():
    """Upload a fabric sample image. Returns a sample_id for use in recolor."""
    if "sample" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["sample"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    data = file.read()
    sample_id = hashlib.md5(data).hexdigest()[:12]
    ext = Path(file.filename).suffix.lower() or ".jpg"
    sample_path = UPLOAD_DIR / f"{sample_id}{ext}"
    sample_path.write_bytes(data)

    # Return a small preview
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((200, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    preview_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "sample_id": sample_id,
        "ext": ext,
        "preview": preview_b64,
    })


@app.route("/api/recolor-stream")
def recolor_stream():
    """SSE endpoint: apply uploaded fabric sample to all template images."""
    sample_id = request.args.get("sample_id", "").strip()
    sample_ext = request.args.get("ext", ".jpg").strip()

    if not sample_id:
        return jsonify({"error": "No sample_id provided"}), 400

    sample_path = UPLOAD_DIR / f"{sample_id}{sample_ext}"
    if not sample_path.exists():
        return jsonify({"error": "Sample not found. Upload again."}), 404

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500

    sample_bytes = sample_path.read_bytes()
    sample_mime = "image/jpeg" if sample_ext in (".jpg", ".jpeg") else "image/png"

    output_dir = OUTPUT_BASE / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    def generate():
        filenames = sorted(IMAGE_PROMPTS.keys())
        total = len(filenames)

        for i, filename in enumerate(filenames):
            yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': total, 'filename': filename, 'status': 'processing'})}\n\n"

            input_path = INPUT_DIR / filename
            if not input_path.exists():
                yield f"data: {json.dumps({'type': 'result', 'index': i, 'filename': filename, 'status': 'error', 'message': 'Source not found'})}\n\n"
                continue

            target = IMAGE_PROMPTS[filename]["target"]
            prompt = build_prompt(target)

            try:
                img_bytes = recolor_image(client, sample_bytes, sample_mime, input_path, prompt)

                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode == "RGBA":
                        img = img.convert("RGB")

                    out_name = filename.replace("WHITE", sample_id)
                    out_path = output_dir / out_name
                    img.save(out_path, "JPEG", quality=95)

                    thumb = img.copy()
                    thumb.thumbnail((400, 400))
                    buf = io.BytesIO()
                    thumb.save(buf, "JPEG", quality=80)
                    b64 = base64.b64encode(buf.getvalue()).decode()

                    yield f"data: {json.dumps({'type': 'result', 'index': i, 'filename': filename, 'status': 'ok', 'output': out_name, 'preview': b64})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', 'index': i, 'filename': filename, 'status': 'error', 'message': 'No image returned by model'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'result', 'index': i, 'filename': filename, 'status': 'error', 'message': str(e)})}\n\n"

            time.sleep(1)

        yield f"data: {json.dumps({'type': 'done', 'output_dir': str(output_dir)})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/output/<path:filepath>")
def serve_output(filepath):
    return send_from_directory(str(OUTPUT_BASE), filepath)


FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bedding Fabric Recolor Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }

  .header { background: #1a1a1a; border-bottom: 1px solid #333; padding: 20px 32px; }
  .header h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; }
  .controls { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }

  /* Upload area */
  .upload-area { display: flex; align-items: center; gap: 12px; }
  .upload-box { width: 80px; height: 80px; border: 2px dashed #444; border-radius: 10px; display: flex; align-items: center; justify-content: center; cursor: pointer; overflow: hidden; transition: border-color 0.2s; background: #222; }
  .upload-box:hover { border-color: #4f46e5; }
  .upload-box img { width: 100%; height: 100%; object-fit: cover; border-radius: 8px; }
  .upload-box .placeholder { color: #666; font-size: 11px; text-align: center; padding: 4px; line-height: 1.3; }
  .upload-label { font-size: 13px; color: #aaa; max-width: 160px; }

  .btn { padding: 10px 24px; border-radius: 8px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: #4f46e5; color: white; }
  .btn-primary:hover { background: #4338ca; }
  .btn-primary:disabled { background: #333; color: #666; cursor: not-allowed; }

  .progress-bar-container { width: 100%; max-width: 400px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; display: none; }
  .progress-bar { height: 100%; background: #4f46e5; border-radius: 3px; transition: width 0.3s; width: 0%; }
  .status-text { font-size: 13px; color: #888; min-width: 200px; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; padding: 24px 32px; }
  .card { background: #1a1a1a; border-radius: 12px; overflow: hidden; border: 1px solid #2a2a2a; transition: border-color 0.2s; }
  .card.processing { border-color: #4f46e5; }
  .card.done { border-color: #22c55e; }
  .card.error { border-color: #ef4444; }
  .card-header { padding: 12px 16px; font-size: 13px; font-weight: 500; color: #999; border-bottom: 1px solid #2a2a2a; display: flex; justify-content: space-between; align-items: center; }
  .card-status { font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .card-status.waiting { background: #333; color: #888; }
  .card-status.processing { background: #4f46e520; color: #818cf8; }
  .card-status.done { background: #22c55e20; color: #4ade80; }
  .card-status.error { background: #ef444420; color: #f87171; }
  .card-images { display: grid; grid-template-columns: 1fr 1fr; }
  .card-images.single { grid-template-columns: 1fr; }
  .card-img-wrap { position: relative; aspect-ratio: 4/3; overflow: hidden; background: #111; }
  .card-img-wrap img { width: 100%; height: 100%; object-fit: cover; }
  .card-img-label { position: absolute; bottom: 6px; left: 6px; font-size: 10px; background: #000a; padding: 2px 6px; border-radius: 3px; color: #ccc; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #4f46e540; border-top-color: #818cf8; border-radius: 50%; animation: spin 0.8s linear infinite; margin: auto; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .card-img-wrap .spinner-overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: #0008; }
</style>
</head>
<body>

<div class="header">
  <h1>Bedding Fabric Recolor Tool</h1>
  <div class="controls">
    <div class="upload-area">
      <div class="upload-box" id="uploadBox" onclick="document.getElementById('fileInput').click()">
        <div class="placeholder" id="uploadPlaceholder">Click to upload fabric sample</div>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleUpload(this)">
      <div class="upload-label" id="uploadLabel">Upload a fabric sample image (solid color or pattern)</div>
    </div>
    <button class="btn btn-primary" id="recolorBtn" onclick="startRecolor()" disabled>Apply Fabric</button>
    <div class="progress-bar-container" id="progressContainer">
      <div class="progress-bar" id="progressBar"></div>
    </div>
    <div class="status-text" id="statusText"></div>
  </div>
</div>

<div class="grid" id="imageGrid"></div>

<script>
let images = [];
let currentSample = null; // { sample_id, ext, preview }

async function loadImages() {
  const res = await fetch("/api/images");
  images = await res.json();
  renderGrid();
}

function renderGrid() {
  const grid = document.getElementById("imageGrid");
  grid.innerHTML = images.map(img => {
    const id = css(img.filename);
    return `
    <div class="card" id="card-${id}">
      <div class="card-header">
        <span>${img.desc}</span>
        <span class="card-status waiting" id="status-${id}">Waiting</span>
      </div>
      <div class="card-images single" id="imgs-${id}">
        <div class="card-img-wrap">
          <img src="/api/preview/${encodeURIComponent(img.filename)}" alt="${img.filename}">
          <span class="card-img-label">Original</span>
        </div>
      </div>
    </div>`;
  }).join("");
}

function css(name) {
  return name.replace(/[^a-zA-Z0-9]/g, "_");
}

async function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("sample", file);

  document.getElementById("uploadLabel").textContent = "Uploading...";
  document.getElementById("recolorBtn").disabled = true;

  try {
    const res = await fetch("/api/upload-sample", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      document.getElementById("uploadLabel").textContent = data.error;
      return;
    }

    currentSample = data;
    document.getElementById("uploadBox").innerHTML = `<img src="data:image/jpeg;base64,${data.preview}">`;
    document.getElementById("uploadLabel").textContent = file.name;
    document.getElementById("recolorBtn").disabled = false;
  } catch (e) {
    document.getElementById("uploadLabel").textContent = "Upload failed: " + e.message;
  }
}

function startRecolor() {
  if (!currentSample) {
    alert("Please upload a fabric sample first.");
    return;
  }

  const btn = document.getElementById("recolorBtn");
  btn.disabled = true;
  btn.textContent = "Processing...";

  const progressContainer = document.getElementById("progressContainer");
  const progressBar = document.getElementById("progressBar");
  const statusText = document.getElementById("statusText");
  progressContainer.style.display = "block";
  progressBar.style.width = "0%";

  // Reset all cards
  images.forEach(img => {
    const id = css(img.filename);
    document.getElementById("card-" + id).className = "card";
    document.getElementById("status-" + id).className = "card-status waiting";
    document.getElementById("status-" + id).textContent = "Waiting";
    const imgsDiv = document.getElementById("imgs-" + id);
    imgsDiv.className = "card-images single";
    const wraps = imgsDiv.querySelectorAll(".card-img-wrap");
    while (wraps.length > 1) wraps[wraps.length - 1].remove();
  });

  const params = new URLSearchParams({
    sample_id: currentSample.sample_id,
    ext: currentSample.ext,
  });
  const evtSource = new EventSource(`/api/recolor-stream?${params}`);

  evtSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "progress") {
      const id = css(data.filename);
      document.getElementById("card-" + id).className = "card processing";
      const st = document.getElementById("status-" + id);
      st.className = "card-status processing";
      st.textContent = "Processing...";
      statusText.textContent = `Processing ${data.index + 1}/${data.total}: ${data.filename}`;
      progressBar.style.width = ((data.index) / data.total * 100) + "%";

      const imgsDiv = document.getElementById("imgs-" + id);
      if (imgsDiv.querySelectorAll(".card-img-wrap").length === 1) {
        imgsDiv.className = "card-images";
        const wrap = document.createElement("div");
        wrap.className = "card-img-wrap";
        wrap.innerHTML = '<div class="spinner-overlay"><div class="spinner"></div></div><span class="card-img-label">Recolored</span>';
        imgsDiv.appendChild(wrap);
      }
    }

    if (data.type === "result") {
      const id = css(data.filename);
      const card = document.getElementById("card-" + id);
      const st = document.getElementById("status-" + id);
      const imgsDiv = document.getElementById("imgs-" + id);

      if (data.status === "ok") {
        card.className = "card done";
        st.className = "card-status done";
        st.textContent = "Done";
        const wraps = imgsDiv.querySelectorAll(".card-img-wrap");
        if (wraps.length > 1) {
          wraps[1].innerHTML = `<img src="data:image/jpeg;base64,${data.preview}" alt="Recolored"><span class="card-img-label">Recolored</span>`;
        }
      } else {
        card.className = "card error";
        st.className = "card-status error";
        st.textContent = "Error";
        const wraps = imgsDiv.querySelectorAll(".card-img-wrap");
        if (wraps.length > 1) {
          wraps[1].innerHTML = `<div style="color:#f87171;padding:16px;font-size:12px;">${data.message}</div><span class="card-img-label">Error</span>`;
        }
      }

      progressBar.style.width = ((data.index + 1) / images.length * 100) + "%";
    }

    if (data.type === "done") {
      evtSource.close();
      btn.disabled = false;
      btn.textContent = "Apply Fabric";
      statusText.textContent = `Done! Saved to ${data.output_dir}`;
      progressBar.style.width = "100%";
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    btn.disabled = false;
    btn.textContent = "Apply Fabric";
    statusText.textContent = "Connection error. Check console.";
  };
}

loadImages();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set. Set it in .env")
        print()
    print(f"Input images: {INPUT_DIR}")
    print(f"Output dir:   {OUTPUT_BASE}")
    print()
    app.run(debug=True, port=5000)
