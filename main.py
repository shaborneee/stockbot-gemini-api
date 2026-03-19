import io
import json
import logging
import os
import re
import sys
import threading
from datetime import date, datetime, timedelta
from flask import Flask, request, jsonify
import google.genai as genai
from dotenv import load_dotenv
import PIL.Image
import requests
import time
from typing import Any

# Load environment variables from .env before creating clients.
load_dotenv()

# Create the Flask app instance used by all routes.
app = Flask(__name__)

# Configure logging to stream detailed runtime logs to stdout.
# Ensure Flask application logs are emitted to STDOUT.
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
app.logger.handlers.clear()
app.logger.addHandler(stdout_handler)
app.logger.setLevel(logging.INFO)
app.logger.propagate = False

# Initialize Gemini client from environment configuration.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Service configuration constants.
# Backend ingestion endpoint
INGEST_URL = "https://stockbot-api-yu48.onrender.com/api/inventory/ingestion/classification/"

# Reduced from 512 → 256 for faster upload + inference
_GEMINI_MAX_DIM = 256

# bot_id must be an integer for your backend
BOT_ID = int(os.getenv("BOT_ID"))


# Helper: normalize Gemini/HTTP exceptions to an HTTP-like status code.
def _extract_gemini_error_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code

    response = getattr(error, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    match = re.search(r"\b(\d{3})\b", str(error))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    return None


# Helper: sanitize and enforce a valid expires_at date format and range.
def _normalize_expires_at(expires_at: Any) -> str | None:
    if not expires_at:
        return None

    expires_at = str(expires_at).strip()
    if not expires_at or expires_at.lower() == "null":
        return None

    try:
        parsed = datetime.strptime(expires_at, "%Y-%m-%d").date()
    except ValueError:
        return None

    today = date.today()
    if parsed <= today:
        return (today + timedelta(days=1)).isoformat()

    return parsed.isoformat()


# Helper: optimize incoming image size/encoding before sending to Gemini.
def _preprocess_image(image_file) -> PIL.Image.Image:
    """
    Open, convert to RGB, resize to _GEMINI_MAX_DIM, and re-encode as a
    lean JPEG in memory. Returns a PIL Image ready for Gemini.
    """
    img = PIL.Image.open(image_file.stream)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if max(img.size) > _GEMINI_MAX_DIM:
        resample = PIL.Image.BILINEAR if max(img.size) > 300 else PIL.Image.NEAREST
        img = img.copy()
        img.thumbnail((_GEMINI_MAX_DIM, _GEMINI_MAX_DIM), resample)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return PIL.Image.open(buf)


# Main inference endpoint: classify grocery image and queue backend ingestion.
@app.route("/detect", methods=["POST"])
def detect():
    # Validate request payload includes an image file.
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_name = image_file.filename

    # Decode and resize image for faster, more stable model calls.
    try:
        img = _preprocess_image(image_file)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    today_str = date.today().isoformat()

    prompt = (
        f"Identify the single most prominent grocery item in this image.\n"
        f"Return ONLY valid JSON, no extra text, no markdown.\n\n"

        f"CLASSIFICATION RULES:\n"
        f"- Read ALL visible text on the packaging (brand, product name, variant, percentage, flavour).\n"
        f"- Build the classification as: '[Brand] [full product descriptor]'.\n"
        f"  Examples: 'Silk 2% Almond Milk', 'Heinz Tomato Ketchup', 'Tropicana Original Orange Juice'.\n"
        f"- If no brand is visible, use just the generic descriptor, e.g. 'Whole Milk'.\n"
        f"- For fresh produce (fruits, vegetables): also assess ripeness visually.\n"
        f"  If the item looks overripe, bruised, or close to spoiling, append '(overripe)' to the name.\n"
        f"  Example: 'Banana (overripe)', 'Avocado (overripe)'.\n"
        f"- Category must be one of: fresh produce, dairy, meat, baked goods, "
        f"canned goods, pantry staples, frozen items, snack foods, beverages.\n"
        f"- If unidentifiable or outside these categories: classification = 'unknown', expires_at = null.\n\n"

        f"EXPIRY RULES:\n"
        f"- Today: {today_str}. expires_at must be yyyy-mm-dd and strictly after today, or null.\n"
        f"- Use these baselines from today:\n"
        f"  Fresh produce (normal): 3-7d | Fresh produce (overripe): 1-2d\n"
        f"  Dairy: 7-14d | Meat: 3-5d | Baked goods: 3-7d\n"
        f"  Canned goods: 1-2 years | Pantry staples: 6-12mo | Frozen: 6-12mo\n"
        f"- If unsure, set expires_at to null.\n\n"

        f'Return exactly: {{"classification": "...", "expires_at": "yyyy-mm-dd or null"}}'
    )

    classification = "unknown"
    expires_at = None
    gemini_time = 0.0
    gemini_error_code = None

    # Call Gemini and parse the structured JSON result.
    try:
        gemini_start = time.time()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img],
            config={"temperature": 0.1},  # lower = more deterministic/faster
        )
        gemini_time = (time.time() - gemini_start) * 1000

        raw = (response.text or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
        except Exception:
            result = {}

        classification = str(result.get("classification", "")).strip() or "unknown"
        expires_at = result.get("expires_at")

        if classification.lower() == "unknown":
            classification = "unknown"
            expires_at = None

        expires_at = _normalize_expires_at(expires_at)
    except Exception as e:
        # Fail safe to unknown when Gemini is unavailable or rate-limited.
        gemini_error_code = _extract_gemini_error_code(e)
        app.logger.warning(
            "gemini_fallback_unknown image_id=%s gemini_error_code=%s reason=%s",
            image_name,
            gemini_error_code,
            str(e),
        )

    # Build payload for backend ingestion service.
    ingest_payload = {
        "bot_id": BOT_ID,
        "image_id": str(image_name),
        "classification": classification,
        "expires_at": expires_at,
    }

    # Background ingestion worker keeps /detect latency low.
    def _ingest_async(payload: dict, name: str) -> None:
        ingest_start = time.time()
        try:
            r = requests.post(
                INGEST_URL,
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            ingest_time = (time.time() - ingest_start) * 1000
            app.logger.info("[INGEST] %s -> status=%s (%.0f ms)", name, r.status_code, ingest_time)
        except Exception as e:
            ingest_time = (time.time() - ingest_start) * 1000
            app.logger.warning("[INGEST] error for %s after %.0f ms: %s", name, ingest_time, str(e))

    threading.Thread(
        target=_ingest_async,
        args=(ingest_payload, image_name),
        daemon=True,
    ).start()

    # Track async ingestion state in API response.
    ingest_status = {
        "sent": True,
        "queued": True,
    }

    # Emit detailed request summary logs for monitoring and debugging.
    app.logger.info("=" * 70)
    app.logger.info("[DETECT] Request Summary")
    app.logger.info("file_name: %s", image_name)
    app.logger.info("classification: %s", classification)
    app.logger.info("expires_at: %s", expires_at)
    app.logger.info("gemini_error_code: %s", gemini_error_code)
    app.logger.info("gemini_time_ms: %.2f", gemini_time)
    app.logger.info("ingest_payload:\n%s", json.dumps(ingest_payload, indent=2))
    app.logger.info("backend_ingestion: queued")
    app.logger.info("=" * 70)

    # Build final response body for the caller.
    api_response = {
        "bot_id": BOT_ID,
        "image_id": image_name,
        "classification": classification,
        "expires_at": expires_at,
        "backend_ingestion": ingest_status,
    }

    # Return JSON response with classification and async ingestion status.
    return app.response_class(
        response=json.dumps(api_response, indent=2),
        status=200,
        mimetype="application/json",
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)