import json
from flask import Flask, request, jsonify
import google.genai as genai
from dotenv import load_dotenv
import os
import PIL.Image
from datetime import datetime, timedelta

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
bot_id = os.getenv("BOT_ID", "stockbot")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_name = image_file.filename

    img = PIL.Image.open(image_file.stream)

    prompt = """Look at this image and identify the single most prominent grocery item.
    Do NOT include brand names and only include generic item names.
    Calculate the expiration date based on when this item would typically spoil from today.
    Return ONLY a valid JSON object in this exact format, no extra text:
    {
        "classification": "classified item name only",
        "expires_at": "yyyy-mm-dd date format"
    }"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )

    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["image_id"] = image_name
    result["bot_id"] = bot_id

    print("\n" + "="*50)
    print(json.dumps(result, indent=2))
    print("="*50 + "\n")

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)