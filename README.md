# StockBot - AI Grocery Item Detector

StockBot is a Flask-based API that uses Google's Gemini 2.5 Flash model to analyze images of grocery items and provide information about their shelf life and storage recommendations.

## Features

- **Image Analysis**: Upload an image of a grocery item and get instant identification
- **Expiration Date Prediction**: Calculates estimated expiration dates in `yyyy-mm-dd` format
- **Generic Classification**: Returns only the generic item name (e.g., milk, bread, cheese)
- **JSON Response**: Structured data output with item classification, expiration date, and metadata

## What the Code Does

The API provides a `/detect` endpoint that:
1. Accepts an image file (POST request)
2. Sends the image to Google's Gemini 2.5 Flash model
3. Uses a custom prompt to identify the grocery item
4. Returns a JSON response with:
   - `classification`: Generic item name
   - `expires_at`: Predicted expiration date (yyyy-mm-dd format)
   - `image_id`: Original filename
   - `bot_id`: Bot identifier

## Installation

### Prerequisites
- Python 3.8+
- pip
- A Google Gemini API key

### Local Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd StockBot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   
   On Windows (PowerShell):
   ```bash
   .\venv\Scripts\Activate.ps1
   ```
   
   On Windows (CMD):
   ```bash
   venv\Scripts\activate.bat
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a `.env` file** in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   BOT_ID=stockbot
   ```

6. **Run the server**:
   ```bash
   python main.py
   ```
   
   The server will start on `http://localhost:5000`

## Usage

### Test with cURL

```bash
curl -X POST -F "image=@photos/soup.jpg" http://localhost:5000/detect
```

Place your image files in the `photos/` folder. Replace `soup.jpg` with your actual image filename.

### Example Response

```json
{
  "classification": "tomato soup",
  "expires_at": "2026-03-12",
  "image_id": "soup.jpg",
  "bot_id": "stockbot"
}
```

## API Endpoints

### POST `/detect`

**Request**:
- Content-Type: `multipart/form-data`
- Body: Image file with key `image`

**Response** (200 OK):
- JSON object with classification, expiration date, and metadata

**Error Response** (400 Bad Request):
```json
{
  "error": "No image file provided"
}
```

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

## Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and add it to your `.env` file

