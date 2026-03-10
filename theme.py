import os
import io
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "WHITE" / "resized"
OUTPUT_BASE = BASE_DIR / "output" / "themes"

# Source images to generate themed versions for
SOURCE_IMAGES = [
    "WHITE FIT 1.jpg",
    "WHITE QLT 1.jpg",
    "WHITE 6.jpg",
]

# Theme definitions - each theme describes the room style
# The bed, bedding color, and position must stay identical
THEMES = {
    "playful": {
        "name": "Child Room / Playful",
        "room_description": (
            "a cheerful children's bedroom with colorful pastel walls, "
            "fun wall decals or stickers (stars, animals, clouds), "
            "soft warm lighting, plush toys on shelves, "
            "a cozy reading nook, playful patterned rug on the floor, "
            "and child-friendly furniture in bright colors. "
            "The room should feel warm, whimsical, and inviting for a child."
        ),
    },
    "modern": {
        "name": "Modern",
        "room_description": (
            "a sleek modern minimalist bedroom with clean lines, "
            "neutral tones (white, grey, black accents), "
            "concrete or light oak flooring, "
            "contemporary pendant lighting, "
            "floor-to-ceiling windows with sheer curtains, "
            "a simple low-profile nightstand with a designer lamp, "
            "and minimal wall art. "
            "The room should feel airy, sophisticated, and contemporary."
        ),
    },
}


def build_theme_prompt(theme_key: str) -> str:
    """Build the Gemini prompt for changing the room theme."""
    theme = THEMES[theme_key]
    return (
        f"Change ONLY the room/background/environment in this image to {theme['room_description']} "
        f"\n\nIMPORTANT RULES:\n"
        f"- Keep the bed in the EXACT same position, angle, and size\n"
        f"- Keep the bedding, pillows, sheets, blankets EXACTLY the same color and style\n"
        f"- Keep the bed frame/headboard the same or similar style\n"
        f"- ONLY change the surrounding room: walls, floor, furniture (nightstands, shelves, decor), lighting, windows\n"
        f"- The result must look like a professional product photo\n"
        f"- Maintain realistic lighting and shadows on the bed that match the new room"
    )


def generate_themed_image(client, image_path: Path, prompt: str) -> bytes | None:
    """Send image to Gemini for room theme change. Returns image bytes or None."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
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


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Add it to .env")
        return

    client = genai.Client(api_key=api_key)

    total_tasks = len(THEMES) * len(SOURCE_IMAGES)
    current = 0

    for theme_key, theme_info in THEMES.items():
        theme_dir = OUTPUT_BASE / theme_key
        theme_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_theme_prompt(theme_key)

        print(f"\n{'='*60}")
        print(f"  Theme: {theme_info['name']}")
        print(f"  Output: {theme_dir}")
        print(f"{'='*60}")

        for filename in SOURCE_IMAGES:
            current += 1
            input_path = INPUT_DIR / filename

            if not input_path.exists():
                print(f"  [{current}/{total_tasks}] SKIP - {filename} not found")
                continue

            print(f"  [{current}/{total_tasks}] Processing {filename}...")

            try:
                img_bytes = generate_themed_image(client, input_path, prompt)

                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode == "RGBA":
                        img = img.convert("RGB")

                    out_name = filename.replace("WHITE", theme_key.upper())
                    out_path = theme_dir / out_name
                    img.save(out_path, "JPEG", quality=95)
                    print(f"           -> Saved: {out_path}")
                else:
                    print(f"           -> ERROR: No image returned by model")

            except Exception as e:
                print(f"           -> ERROR: {e}")

            time.sleep(2)  # avoid rate limits

    print(f"\nDone! Check {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
