import json
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta
from flask import Flask, request, jsonify
import google.genai as genai
from dotenv import load_dotenv
import PIL.Image
import requests
import time

load_dotenv()

app = Flask(__name__)

# Ensure Flask application logs are emitted to STDOUT.
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
app.logger.handlers.clear()
app.logger.addHandler(stdout_handler)
app.logger.setLevel(logging.INFO)
app.logger.propagate = False

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Backend ingestion endpoint
INGEST_URL = "https://stockbot-api-yu48.onrender.com/api/inventory/ingestion/classification/"

# bot_id must be an integer for your backend
BOT_ID = int(os.getenv("BOT_ID"))


def _extract_gemini_error_code(error: Exception) -> int | None:
    # Prefer explicit SDK fields when available.
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

    # Fallback: parse first 3-digit HTTP-like code from message.
    match = re.search(r"\b(\d{3})\b", str(error))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    return None


def _normalize_expires_at(expires_at: str) -> str:
    if not expires_at:
        return ""

    try:
        parsed = datetime.strptime(expires_at, "%Y-%m-%d").date()
    except ValueError:
        return ""

    today = date.today()
    if parsed <= today:
        return (today + timedelta(days=1)).isoformat()

    return parsed.isoformat()


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_name = image_file.filename

    try:
        img = PIL.Image.open(image_file.stream)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    today_str = date.today().isoformat()
    prompt = f"""Look at this image and identify the single most prominent grocery item.
Do NOT include brand names and only include a generic item name.
The item MUST belong to one of these categories: fresh produce, dairy, meat, baked goods, canned goods, pantry staples, frozen items, snack foods, beverages.
If the item cannot be identified, or if it does not belong to one of those categories, set classification to "unknown" and expires_at to empty string.
Today's date is {today_str}.
Based on the item type, calculate the realistic expiration date (when it would typically spoil from today).
expires_at MUST be a real calendar date in yyyy-mm-dd format and MUST be strictly after {today_str}.
Never return today's date or any past date.
If you are unsure of the date, set expires_at to empty string.
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
    "expires_at": "yyyy-mm-dd date format (must be after today)"
}}"""

    classification = "unknown"
    expires_at = ""
    gemini_time = 0.0
    gemini_error_code = None
    response = None

    # Gemini failures (e.g., 429 quota) should not fail the request.
    try:
        gemini_start = time.time()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img]
        )
        gemini_time = (time.time() - gemini_start) * 1000

        raw = (response.text or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
        except Exception:
            result = {}

        classification = str(result.get("classification", "")).strip() or "unknown"
        expires_at = str(result.get("expires_at", "")).strip()

        if classification.lower() == "unknown":
            classification = "unknown"
            expires_at = ""

        expires_at = _normalize_expires_at(expires_at)
    except Exception as e:
        gemini_error_code = _extract_gemini_error_code(e)
        app.logger.warning(
            "gemini_fallback_unknown image_id=%s gemini_error_code=%s reason=%s",
            image_name,
            gemini_error_code,
            str(e),
        )

    # Build payload for backend ingestion
    ingest_payload = {
        "bot_id": BOT_ID,
        "image_id": str(image_name),
        "classification": classification,
        "expires_at": expires_at
    }

    # POST to backend
    ingest_status = {
        "sent": False,
        "status_code": None,
        "response_text": None,
        "error": None
    }

    ingest_start = time.time()
    try:
        r = requests.post(
            INGEST_URL,
            json=ingest_payload,
            timeout=15,
            headers={"Content-Type": "application/json"}
        )
        ingest_status["sent"] = True
        ingest_status["status_code"] = r.status_code
        ingest_status["response_text"] = r.text
    except Exception as e:
        ingest_status["error"] = str(e)
    ingest_time = (time.time() - ingest_start) * 1000

    app.logger.info("=" * 70)
    app.logger.info("[DETECT] Request Summary")
    app.logger.info("file_name: %s", image_name)
    app.logger.info("classification: %s", classification)
    app.logger.info("gemini_error_code: %s", gemini_error_code)
    app.logger.info("gemini_time_ms: %.2f", gemini_time)
    app.logger.info("backend_roundtrip_ms: %.2f", ingest_time)
    app.logger.info("ingest_payload:\n%s", json.dumps(ingest_payload, indent=2))
    app.logger.info("backend_sent: %s", ingest_status.get("sent"))
    app.logger.info("backend_status_code: %s", ingest_status.get("status_code"))
    app.logger.info("backend_response_text: %s", ingest_status.get("response_text"))
    app.logger.info("backend_error: %s", ingest_status.get("error"))
    app.logger.info("=" * 70)

    # Response to client
    api_response = {
        "bot_id": BOT_ID,
        "image_id": image_name,
        "classification": classification,
        "expires_at": expires_at,
        "backend_ingestion": ingest_status
    }

    return app.response_class(
        response=json.dumps(api_response, indent=2),
        status=200,
        mimetype="application/json",
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)