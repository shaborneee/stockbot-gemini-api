# StockBot - AI Grocery Item Detector

Flask API that receives an image, sends it to Gemini 2.5 Flash for grocery classification, then posts the result to your ingestion backend.

## How To Run Locally

### 1) Prerequisites
- Python 3.8+
- pip
- Gemini API key

### 2) Setup
```bash
python -m venv .venv
```

Windows PowerShell:
```bash
.\.venv\Scripts\Activate.ps1
```

Windows CMD:
```bash
.venv\Scripts\activate.bat
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
BOT_ID=1
```

Notes:
- `BOT_ID` must be an integer because `main.py` casts it with `int(...)`.

### 3) Start server
```bash
python main.py
```

Server URL:
- `http://localhost:5000`

### 4) Test the endpoint locally

Multipart upload:
```bash
curl -X POST -F "image=@photos/soup.jpg" http://localhost:5000/detect
```

Raw image bytes (ESP32-style):
```bash
curl -X POST \
  -H "Content-Type: image/jpeg" \
  -H "X-Filename: esp32_test.jpg" \
  --data-binary "@photos/soup.jpg" \
  http://localhost:5000/detect
```

### 5) Response format

Success (`200`):
```json
{
  "bot_id": 1,
  "image_id": "esp32_test.jpg",
  "classification": "milk",
  "expires_at": "2026-03-24",
  "backend_ingestion": {
    "sent": true,
    "status_code": 201,
    "response_text": "...",
    "error": null
  }
}
```

Error examples:

No image provided (`400`):
```json
{
  "error": "No image file or ESP32-CAM image bytes provided"
}
```

Gemini invalid JSON (`502`):
```json
{
  "error": "Gemini returned invalid JSON",
  "raw": "..."
}
```

## How The AI Server And ESP32 Camera Works

### End-to-end flow
1. ESP32-CAM captures a photo.
2. ESP32 sends the image to `POST /detect`.
3. Flask API loads the image and sends it to Gemini 2.5 Flash with the prompt rules.
4. API parses `classification` and `expires_at` from Gemini output.
5. API sends ingestion payload to backend:
   - `bot_id`
   - `image_id`
   - `classification`
   - `expires_at`
6. API returns response to client with backend ingestion status.

### What gets logged
For each request, the server logs a readable summary with:
- file name
- classification
- gemini confidence (or `N/A` if unavailable)
- gemini classification time (ms)
- backend roundtrip time (ms)
- ingestion payload

### ESP32-CAM local instructions

If your Flask server is running on your laptop and ESP32 is on the same Wi-Fi:
- Use laptop local IP (for example `192.168.1.50`), not `localhost`.
- Endpoint becomes: `http://192.168.1.50:5000/detect`

### Connect AI-Thinker ESP32-CAM In Arduino IDE

1. Install Arduino IDE (2.x recommended).
2. Open File > Preferences.
3. Add this URL to Additional Boards Manager URLs:
  - `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
4. Open Tools > Board > Boards Manager, search for `esp32`, then install Espressif ESP32.
5. Open `esp32_cam_module.ino`.
6. In Tools > Board, select `AI Thinker ESP32-CAM`.
7. Select the correct COM port for your camera/USB-serial adapter.

Upload steps (standard ESP32-CAM without native USB):
- Connect USB-serial TX -> U0R and RX -> U0T.
- Connect 5V and GND.
- Pull IO0 to GND to enter flashing mode.
- Click Upload.
- After upload, disconnect IO0 from GND and press RST (or power-cycle).

If you have an ESP32-CAM-MB (USB baseboard), plug it in by USB, choose the COM port, upload, then press reset if needed.

### Build And Run The Camera Sketch

1. In `esp32_cam_module.ino`, set:
  - `WIFI_SSID`
  - `WIFI_PASS`
  - `SERVER_URL`
2. Click Verify/Compile in Arduino IDE.
3. Click Upload.
4. Open Serial Monitor at 115200 baud to watch motion/upload logs.

### Power Options After Upload

- Powered by laptop USB: keep the camera plugged into your laptop and it will keep detecting motion and sending pictures.
- Powered by external 5V source: use a stable 5V battery/power bank and the camera will keep detecting motion and sending pictures.

Power tip:
- ESP32-CAM is sensitive to voltage drops. Use a stable 5V source with enough current for camera + Wi-Fi bursts.
```