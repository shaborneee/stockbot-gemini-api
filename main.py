import json
import os
from flask import Flask, request, jsonify
import google.genai as genai
from dotenv import load_dotenv
import PIL.Image
import requests
import time

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Backend ingestion endpoint
INGEST_URL = "https://stockbot-api-yu48.onrender.com/api/inventory/ingestion/classification/"

# bot_id must be an integer for your backend
BOT_ID = int(os.getenv("BOT_ID"))


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

    prompt = """Look at this image and identify the single most prominent grocery item.
Do NOT include brand names and only include generic item names.
Calculate the expiration date based on when this item would typically spoil from today.
Return ONLY a valid JSON object in this exact format, no extra text:
{
  "classification": "classified item name only",
  "expires_at": "yyyy-mm-dd date format"
}"""

    # Time Gemini classification
    gemini_start = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )
    gemini_time = (time.time() - gemini_start) * 1000

    raw = (response.text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Parse Gemini JSON safely
    try:
        result = json.loads(raw)
    except Exception:
        return jsonify({
            "error": "Gemini returned invalid JSON",
            "raw": raw
        }), 502

    # Validate required keys
    classification = str(result.get("classification", "")).strip()
    expires_at = str(result.get("expires_at", "")).strip()

    if not classification or not expires_at:
        return jsonify({
            "error": "Missing classification or expires_at from Gemini",
            "gemini_result": result
        }), 502

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

    # Response to client
    api_response = {
        "bot_id": BOT_ID,
        "image_id": image_name,
        "classification": classification,
        "expires_at": expires_at,
        "backend_ingestion": ingest_status
    }

    print("\n" + "=" * 50)
    print(f"Classification time: {gemini_time:.2f}ms")
    print(f"Backend ingestion time: {ingest_time:.2f}ms")
    print("\nResponse:")
    print(json.dumps(api_response, indent=2))
    print("=" * 50 + "\n")

    return jsonify(api_response), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)