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
THEME_DIR = BASE_DIR / "themes"
OUTPUT_BASE = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Themes ──────────────────────────────────────────────────────
# Each theme has 3 room images. "classy" uses originals from the theme dir.
# The theme files map a canonical key to the actual filename on disk.
THEMES = {
    "classy": {
        "label": "Classy",
        "files": {
            "6":     "WHITE 6.jpg",
            "FIT 1": "WHITE FIT 1.jpg",
            "QLT 1": "WHITE QLT 1.jpg",
        },
    },
    "modern": {
        "label": "Modern",
        "files": {
            "6":     "MODERN 6.jpg",
            "FIT 1": "MODERN FIT 1.jpg",
            "QLT 1": "MODERN QLT 1.jpg",
        },
    },
    "playful": {
        "label": "Playful",
        "files": {
            "6":     "PLAYFUL 6.jpg",
            "FIT 1": "PLAYFUL FIT 1.jpg",
            "QLT 1": "PLAYFUL QLT 1.jpg",
        },
    },
}

# ── Image configs ───────────────────────────────────────────────
# "source": "resized" = always from WHITE/resized
# "source": "theme"   = from output/themes/{theme}/
IMAGE_CONFIGS = [
    {
        "key": "WHITE 10 ABC.jpg",
        "source": "resized",
        "target": "the bolster (the cylindrical pillow)",
        "desc": "Bolster",
    },
    {
        "key": "WHITE 11 APC.jpg",
        "source": "resized",
        "target": "both pillows",
        "desc": "Pillows",
    },
    {
        "key": "WHITE 3.jpg",
        "source": "resized",
        "target": "the entire fabric and button closure area (but not the button itself)",
        "desc": "Fabric & Button",
    },
    {
        "key": "WHITE 4.jpg",
        "source": "resized",
        "target": "only the top-left fabric piece and the fabric tail/loop",
        "desc": "Fabric Corner & Tail",
    },
    {
        "key": "WHITE 5.jpg",
        "source": "resized",
        "target": "only the pillow (the pillowcase fabric)",
        "desc": "Pillow Close-up",
    },
    {
        "key": "6",
        "source": "theme",
        "target": "the entire bed linen including the duvet cover, bed sheet, and all pillows",
        "desc": "Full Bed Set",
    },
    {
        "key": "WHITE 7.jpg",
        "source": "resized",
        "target": "the top part of the bed - the fitted sheet fabric on top of the mattress",
        "desc": "Fitted Sheet Top",
    },
    {
        "key": "WHITE 9.jpg",
        "source": "resized",
        "target": "the bottom-left side elastic/gathered fabric",
        "desc": "Elastic Fabric",
    },
    {
        "key": "FIT 1",
        "source": "theme",
        "target": "the entire bed sheet, all pillows, the bolster, and the blanket",
        "desc": "Fitted Bed Set",
    },
    {
        "key": "QLT 1",
        "source": "theme",
        "target": "the bed duvet/quilt, all pillows, and the blanket",
        "desc": "Quilt Bed Set",
    },
    {
        "key": "WHITE QLT 2.jpg",
        "source": "resized",
        "target": "the entire center bed including the duvet/quilt, all pillows, and blankets",
        "desc": "Quilt Center Bed",
    },
]


def resolve_source_path(cfg: dict, theme: str) -> Path:
    """Get the actual file path for an image config given the selected theme."""
    if cfg["source"] == "resized":
        return INPUT_DIR / cfg["key"]
    else:
        # theme image
        filename = THEMES[theme]["files"][cfg["key"]]
        return THEME_DIR / theme / filename


def build_prompt(target: str) -> str:
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


# ── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return FRONTEND_HTML


@app.route("/api/themes")
def list_themes():
    return jsonify([{"key": k, "label": v["label"]} for k, v in THEMES.items()])


@app.route("/api/images")
def list_images():
    theme = request.args.get("theme", "classy")
    result = []
    for cfg in IMAGE_CONFIGS:
        path = resolve_source_path(cfg, theme)
        result.append({
            "key": cfg["key"],
            "desc": cfg["desc"],
            "source": cfg["source"],
            "exists": path.exists(),
        })
    return jsonify(result)


@app.route("/api/preview/<path:filename>")
def preview_image(filename):
    """Serve from WHITE/resized."""
    return send_from_directory(str(INPUT_DIR), filename)


@app.route("/api/preview-theme/<theme>/<path:key>")
def preview_theme_image(theme, key):
    """Serve a theme-based room image."""
    if theme not in THEMES or key not in THEMES[theme]["files"]:
        return "Not found", 404
    filename = THEMES[theme]["files"][key]
    return send_from_directory(str(THEME_DIR / theme), filename)


@app.route("/api/upload-sample", methods=["POST"])
def upload_sample():
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

    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((200, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    preview_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"sample_id": sample_id, "ext": ext, "preview": preview_b64})


@app.route("/api/recolor-stream")
def recolor_stream():
    sample_id = request.args.get("sample_id", "").strip()
    sample_ext = request.args.get("ext", ".jpg").strip()
    theme = request.args.get("theme", "classy").strip()

    if not sample_id:
        return jsonify({"error": "No sample_id provided"}), 400
    if theme not in THEMES:
        return jsonify({"error": f"Unknown theme: {theme}"}), 400

    sample_path = UPLOAD_DIR / f"{sample_id}{sample_ext}"
    if not sample_path.exists():
        return jsonify({"error": "Sample not found. Upload again."}), 404

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500

    sample_bytes = sample_path.read_bytes()
    sample_mime = "image/jpeg" if sample_ext in (".jpg", ".jpeg") else "image/png"

    output_dir = OUTPUT_BASE / f"{sample_id}_{theme}"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    def generate():
        total = len(IMAGE_CONFIGS)

        for i, cfg in enumerate(IMAGE_CONFIGS):
            display_name = cfg["desc"]
            yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': total, 'key': cfg['key'], 'status': 'processing'})}\n\n"

            input_path = resolve_source_path(cfg, theme)
            if not input_path.exists():
                yield f"data: {json.dumps({'type': 'result', 'index': i, 'key': cfg['key'], 'status': 'error', 'message': f'Source not found: {input_path.name}'})}\n\n"
                continue

            prompt = build_prompt(cfg["target"])

            try:
                img_bytes = recolor_image(client, sample_bytes, sample_mime, input_path, prompt)

                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode == "RGBA":
                        img = img.convert("RGB")

                    out_name = f"{cfg['desc'].replace(' ', '_')}_{theme}.jpg"
                    out_path = output_dir / out_name
                    img.save(out_path, "JPEG", quality=95)

                    thumb = img.copy()
                    thumb.thumbnail((400, 400))
                    buf = io.BytesIO()
                    thumb.save(buf, "JPEG", quality=80)
                    b64 = base64.b64encode(buf.getvalue()).decode()

                    yield f"data: {json.dumps({'type': 'result', 'index': i, 'key': cfg['key'], 'status': 'ok', 'output': out_name, 'preview': b64})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', 'index': i, 'key': cfg['key'], 'status': 'error', 'message': 'No image returned by model'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'result', 'index': i, 'key': cfg['key'], 'status': 'error', 'message': str(e)})}\n\n"

            time.sleep(1)

        yield f"data: {json.dumps({'type': 'done', 'output_dir': str(output_dir)})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/output/<path:filepath>")
def serve_output(filepath):
    return send_from_directory(str(OUTPUT_BASE), filepath)


# ── Frontend ────────────────────────────────────────────────────

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bedding Fabric Recolor Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #1a1a1a; min-height: 100vh; }

  .header { background: #ffffff; border-bottom: 1px solid #e0e0e0; padding: 20px 32px; }
  .header h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; }
  .controls { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }

  /* Theme selector */
  .theme-selector { display: flex; gap: 8px; }
  .theme-btn { padding: 8px 18px; border-radius: 8px; border: 2px solid #d0d0d0; background: #f0f0f0; color: #666; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .theme-btn:hover { border-color: #aaa; color: #333; }
  .theme-btn.active { border-color: #4f46e5; color: #4f46e5; background: #4f46e510; }

  /* Upload area */
  .upload-area { display: flex; align-items: center; gap: 12px; }
  .upload-box { width: 80px; height: 80px; border: 2px dashed #ccc; border-radius: 10px; display: flex; align-items: center; justify-content: center; cursor: pointer; overflow: hidden; transition: border-color 0.2s; background: #fafafa; }
  .upload-box:hover { border-color: #4f46e5; }
  .upload-box img { width: 100%; height: 100%; object-fit: cover; border-radius: 8px; }
  .upload-box .placeholder { color: #999; font-size: 11px; text-align: center; padding: 4px; line-height: 1.3; }
  .upload-label { font-size: 13px; color: #666; max-width: 160px; }

  .divider { width: 1px; height: 40px; background: #d0d0d0; }

  .btn { padding: 10px 24px; border-radius: 8px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-primary { background: #4f46e5; color: white; }
  .btn-primary:hover { background: #4338ca; }
  .btn-primary:disabled { background: #e0e0e0; color: #999; cursor: not-allowed; }

  .progress-bar-container { width: 100%; max-width: 400px; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; display: none; }
  .progress-bar { height: 100%; background: #4f46e5; border-radius: 3px; transition: width 0.3s; width: 0%; }
  .status-text { font-size: 13px; color: #777; min-width: 200px; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; padding: 24px 32px; }
  .card { background: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e0e0e0; transition: border-color 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .card.processing { border-color: #4f46e5; }
  .card.done { border-color: #16a34a; }
  .card.error { border-color: #dc2626; }
  .card-header { padding: 12px 16px; font-size: 13px; font-weight: 500; color: #666; border-bottom: 1px solid #e8e8e8; display: flex; justify-content: space-between; align-items: center; }
  .card-header .badge { font-size: 10px; padding: 2px 6px; border-radius: 3px; background: #4f46e510; color: #4f46e5; margin-left: 8px; }
  .card-status { font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .card-status.waiting { background: #f0f0f0; color: #999; }
  .card-status.processing { background: #4f46e510; color: #4f46e5; }
  .card-status.done { background: #dcfce7; color: #16a34a; }
  .card-status.error { background: #fee2e2; color: #dc2626; }
  .card-images { display: grid; grid-template-columns: 1fr 1fr; }
  .card-images.single { grid-template-columns: 1fr; }
  .card-img-wrap { position: relative; aspect-ratio: 4/3; overflow: hidden; background: #f0f0f0; }
  .card-img-wrap img { width: 100%; height: 100%; object-fit: cover; }
  .card-img-label { position: absolute; bottom: 6px; left: 6px; font-size: 10px; background: #fffc; padding: 2px 6px; border-radius: 3px; color: #333; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #4f46e530; border-top-color: #4f46e5; border-radius: 50%; animation: spin 0.8s linear infinite; margin: auto; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .card-img-wrap .spinner-overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: #fff8; }
</style>
</head>
<body>

<div class="header">
  <h1>Bedding Fabric Recolor Tool</h1>
  <div class="controls">
    <div class="theme-selector" id="themeSelector"></div>
    <div class="divider"></div>
    <div class="upload-area">
      <div class="upload-box" id="uploadBox" onclick="document.getElementById('fileInput').click()">
        <div class="placeholder" id="uploadPlaceholder">Upload fabric sample</div>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleUpload(this)">
      <div class="upload-label" id="uploadLabel">Upload a fabric sample</div>
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
let allImages = [];
let themes = [];
let selectedTheme = "classy";
let currentSample = null;

async function init() {
  const res = await fetch("/api/themes");
  themes = await res.json();
  renderThemeButtons();
  await loadImages();
}

function renderThemeButtons() {
  const container = document.getElementById("themeSelector");
  container.innerHTML = themes.map(t =>
    `<button class="theme-btn ${t.key === selectedTheme ? 'active' : ''}"
            data-theme="${t.key}" onclick="selectTheme('${t.key}')">${t.label}</button>`
  ).join("");
}

async function selectTheme(key) {
  selectedTheme = key;
  renderThemeButtons();
  await loadImages();
}

async function loadImages() {
  const res = await fetch(`/api/images?theme=${selectedTheme}`);
  allImages = await res.json();
  renderGrid();
}

function previewUrl(img) {
  if (img.source === "theme") {
    return `/api/preview-theme/${selectedTheme}/${encodeURIComponent(img.key)}`;
  }
  return `/api/preview/${encodeURIComponent(img.key)}`;
}

function renderGrid() {
  const grid = document.getElementById("imageGrid");
  grid.innerHTML = allImages.map(img => {
    const id = css(img.key);
    const badge = img.source === "theme" ? `<span class="badge">${selectedTheme}</span>` : "";
    return `
    <div class="card" id="card-${id}">
      <div class="card-header">
        <span>${img.desc}${badge}</span>
        <span class="card-status waiting" id="status-${id}">Waiting</span>
      </div>
      <div class="card-images single" id="imgs-${id}">
        <div class="card-img-wrap">
          <img src="${previewUrl(img)}" alt="${img.desc}">
          <span class="card-img-label">Original</span>
        </div>
      </div>
    </div>`;
  }).join("");
}

function css(name) { return name.replace(/[^a-zA-Z0-9]/g, "_"); }

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
    if (data.error) { document.getElementById("uploadLabel").textContent = data.error; return; }

    currentSample = data;
    document.getElementById("uploadBox").innerHTML = `<img src="data:image/jpeg;base64,${data.preview}">`;
    document.getElementById("uploadLabel").textContent = file.name;
    document.getElementById("recolorBtn").disabled = false;
  } catch (e) {
    document.getElementById("uploadLabel").textContent = "Upload failed: " + e.message;
  }
}

function startRecolor() {
  if (!currentSample) { alert("Please upload a fabric sample first."); return; }

  const btn = document.getElementById("recolorBtn");
  btn.disabled = true;
  btn.textContent = "Processing...";

  const progressContainer = document.getElementById("progressContainer");
  const progressBar = document.getElementById("progressBar");
  const statusText = document.getElementById("statusText");
  progressContainer.style.display = "block";
  progressBar.style.width = "0%";

  // Disable theme switching during processing
  document.querySelectorAll(".theme-btn").forEach(b => b.disabled = true);

  // Reset all cards
  allImages.forEach(img => {
    const id = css(img.key);
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
    theme: selectedTheme,
  });
  const evtSource = new EventSource(`/api/recolor-stream?${params}`);

  evtSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "progress") {
      const id = css(data.key);
      document.getElementById("card-" + id).className = "card processing";
      const st = document.getElementById("status-" + id);
      st.className = "card-status processing";
      st.textContent = "Processing...";
      statusText.textContent = `Processing ${data.index + 1}/${data.total}`;
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
      const id = css(data.key);
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
      progressBar.style.width = ((data.index + 1) / allImages.length * 100) + "%";
    }

    if (data.type === "done") {
      evtSource.close();
      btn.disabled = false;
      btn.textContent = "Apply Fabric";
      statusText.textContent = `Done! Saved to ${data.output_dir}`;
      progressBar.style.width = "100%";
      document.querySelectorAll(".theme-btn").forEach(b => b.disabled = false);
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    btn.disabled = false;
    btn.textContent = "Apply Fabric";
    statusText.textContent = "Connection error. Check console.";
    document.querySelectorAll(".theme-btn").forEach(b => b.disabled = false);
  };
}

init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set. Set it in .env")
        print()
    print(f"Input images: {INPUT_DIR}")
    print(f"Theme images: {THEME_DIR}")
    print(f"Output dir:   {OUTPUT_BASE}")
    print()
    app.run(debug=True, port=5000)
