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


def _encode_jpeg_bytes(img: PIL.Image.Image) -> bytes:
    """Encode a PIL image as JPEG bytes for backend multipart upload."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# Main inference endpoint: classify grocery image and queue backend ingestion.
@app.route("/detect", methods=["POST"])
def detect():
    # Record server-side arrival time for transit calculation.
    server_received_at_ms = time.time() * 1000

    # Validate request payload includes an image file.
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_name = image_file.filename

    # Read timing fields sent by the ESP32.
    # capture_ms: time from motion detected to photo taken on the ESP32.
    # ai_send_timestamp_ms: ESP32 millis() recorded just before the HTTP POST.
    try:
        capture_ms = int(request.form.get("capture_ms", 0))
    except (ValueError, TypeError):
        capture_ms = None

    try:
        ai_send_timestamp_ms = int(request.form.get("ai_send_timestamp_ms", 0))
    except (ValueError, TypeError):
        ai_send_timestamp_ms = None

    # Decode and resize image for faster, more stable model calls.
    try:
        img = _preprocess_image(image_file)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    today_str = date.today().isoformat()

    prompt = f"""Look at this image and identify the single most prominent grocery item.
Do NOT include brand names and only include a generic item name.
The item MUST belong to one of these categories: fresh produce, dairy, meat, baked goods, canned goods, pantry staples, frozen items, snack foods, beverages.
If the item cannot be identified, or if it does not belong to one of those categories, set classification to "unknown" and expires_at to null.
Today's date is {today_str}.
Based on the item type, calculate the realistic expiration date (when it would typically spoil from today).
expires_at MUST be either null or a real calendar date in yyyy-mm-dd format that is strictly after {today_str}.
Never return today's date or any past date.
If you are unsure of the date, set expires_at to null.
Use these shelf life guidelines:
- Fresh produce (fruits, vegetables): 3-7 days
- Dairy (milk, yogurt, cheese): 7-14 days
- Meat (fresh): 3-5 days
- Baked goods: 3-7 days
- Canned goods: 1-2 years
- Pantry staples: 6-12 months
- Frozen items: 6-12 months
Return ONLY a valid JSON object in this exact format, no extra text:
{{
    "classification": "classified item name only, or unknown",
    "expires_at": null
}}"""

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

    # Ensure response contract: unknown or missing date must be returned as JSON null.
    if classification.lower() == "unknown" or not expires_at:
        expires_at = None

    # Build metadata payload for backend ingestion service.
    ingest_form_data = {
        "bot_id": BOT_ID,
        "image_id": str(image_name),
        "classification": classification,
        "expires_at": expires_at,
    }

    jpeg_bytes = _encode_jpeg_bytes(img)
    image_stem = os.path.splitext(str(image_name or "capture"))[0]
    ingest_filename = f"{image_stem}.jpg"
    ingest_files = {
        "image": (ingest_filename, jpeg_bytes, "image/jpeg")
    }

    # Background ingestion worker keeps /detect latency low.
    # Logs how long the DB/backend call takes once it completes.
    def _ingest_async(form_data: dict, files_payload: dict, name: str) -> None:
        ingest_start = time.time()
        try:
            r = requests.post(
                INGEST_URL,
                data=form_data,
                files=files_payload,
                timeout=15,
            )
            ingest_time = (time.time() - ingest_start) * 1000
            app.logger.info(
                "[INGEST] %s -> status=%s | db_ingest_time_ms=%.0f ms",
                name, r.status_code, ingest_time,
            )
        except Exception as e:
            ingest_time = (time.time() - ingest_start) * 1000
            app.logger.warning(
                "[INGEST] error for %s after %.0f ms: %s",
                name, ingest_time, str(e),
            )

    threading.Thread(
        target=_ingest_async,
        args=(ingest_form_data, ingest_files, image_name),
        daemon=True,
    ).start()

    # Track async ingestion state in API response.
    ingest_status = {
        "sent": True,
        "queued": True,
    }

    # Compute ESP32 → server network transit time.
    server_processing_ms = (time.time() * 1000) - server_received_at_ms
    esp32_to_server_ms = server_processing_ms - gemini_time

    # Emit timing summary logs.
    app.logger.info("=" * 70)
    app.logger.info("[DETECT] %s | classification=%s | expires_at=%s", image_name, classification, expires_at)
    app.logger.info("[TIMING] capture_ms=%s ms  (motion detected → photo taken on ESP32)",
                    capture_ms if capture_ms is not None else "N/A")
    app.logger.info("[TIMING] esp32_to_server_ms=%.0f ms  (network upload + server decode)", esp32_to_server_ms)
    app.logger.info("[TIMING] gemini_time_ms=%.0f ms  (Gemini AI inference)", gemini_time)
    app.logger.info("[TIMING] db_ingest_time_ms=see [INGEST] log below  (async, logged when complete)")
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
