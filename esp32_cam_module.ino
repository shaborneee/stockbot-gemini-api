#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <esp_now.h>

// ── WiFi & Server ──────────────────────────────────────────────
const char* WIFI_SSID  = "Fill in Your WiFi SSID";
const char* WIFI_PASS  = "Fill in Your WiFi Password";
const char* SERVER_URL = "https://stockbot-gemini-api.onrender.com/detect";

// ── ESP-NOW peer address ───────────────────────────────────────
uint8_t broadcastAddress[] = "Fill in Your ESP-NOW Peer MAC";

// ── Motion & Capture Settings ──────────────────────────────────
const int   MOTION_COOLDOWN_MS      = 2000;
const int   POST_CAPTURE_LOCKOUT_MS = 10000;
const int   DIFF_THRESHOLD          = 25;
const int   HTTP_TIMEOUT_MS         = 30000;
const float MOTION_PERCENT          = 5.0f;
const unsigned long MOTION_LOG_INTERVAL_MS = 1200;

// ── LED Pins ───────────────────────────────────────────────────
#define FLASH_LED_PIN  4
#define STATUS_LED_PIN 33

// ── Camera Pins (AI-Thinker ESP32-CAM) ────────────────────────
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// ── State Machine ──────────────────────────────────────────────
enum State { IDLE, MOTION_DETECTED, WAITING_FOR_STILL, CAPTURING, LOCKOUT };
State currentState = IDLE;

unsigned long motionStoppedAt = 0;
unsigned long lockoutStart    = 0;
unsigned long lastMotionLogAt = 0;

// ── Buffers ────────────────────────────────────────────────────
// prevFrame = for frame-to-frame motion detection
uint8_t* prevFrame = nullptr;
int      prevFrameSize = 0;

// baselineFrame = fixed baseline captured when motion first stops
uint8_t* baselineFrame = nullptr;
int      baselineSize  = 0;

// ── ESP-NOW tracking ───────────────────────────────────────────
char lastEspNowFilename[250] = {0};

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────
const char* stateToString(State s) {
  switch (s) {
    case IDLE:              return "IDLE";
    case MOTION_DETECTED:   return "MOTION_DETECTED";
    case WAITING_FOR_STILL: return "WAITING_FOR_STILL";
    case CAPTURING:         return "CAPTURING";
    case LOCKOUT:           return "LOCKOUT";
    default:                return "UNKNOWN";
  }
}

void setState(State newState) {
  if (currentState != newState) {
    Serial.printf("[STATE] %s -> %s\n", stateToString(currentState), stateToString(newState));
    currentState = newState;
  }
}

void freeMotionBuffers() {
  free(prevFrame);
  prevFrame = nullptr;
  prevFrameSize = 0;

  free(baselineFrame);
  baselineFrame = nullptr;
  baselineSize = 0;
}

inline uint8_t rgb565ToLuma(const uint8_t* buf, int pixelIndex) {
  uint16_t px = ((uint16_t)buf[pixelIndex * 2] << 8) | buf[pixelIndex * 2 + 1];
  uint8_t r = (px >> 11) & 0x1F;
  uint8_t g = (px >> 5)  & 0x3F;
  uint8_t b = px & 0x1F;
  return (uint8_t)((r * 8 * 299 + g * 4 * 587 + b * 8 * 114) / 1000);
}

// ─────────────────────────────────────────────────────────────
//  ESP-NOW
// ─────────────────────────────────────────────────────────────
void onDataSent(const wifi_tx_info_t* info, esp_now_send_status_t status) {
  Serial.printf("[ESP-NOW] Callback for '%s' -> %s\n",
                lastEspNowFilename[0] ? lastEspNowFilename : "(unknown)",
                status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAIL");
}

void initESPNow() {
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] init FAILED");
    return;
  }

  esp_now_register_send_cb(onDataSent);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, broadcastAddress, 6);
  peer.channel = 0;
  peer.encrypt = false;

  esp_err_t r = esp_now_add_peer(&peer);
  if (r != ESP_OK && r != ESP_ERR_ESPNOW_EXIST) {
    Serial.printf("[ESP-NOW] Failed to add peer (err=%d)\n", r);
  } else {
    Serial.println("[ESP-NOW] Ready.");
  }
}

bool sendFilenameESPNow(const String& filename) {
  memset(lastEspNowFilename, 0, sizeof(lastEspNowFilename));
  filename.toCharArray(lastEspNowFilename, sizeof(lastEspNowFilename));

  esp_err_t r = esp_now_send(
    broadcastAddress,
    (uint8_t*)lastEspNowFilename,
    strlen(lastEspNowFilename) + 1
  );

  if (r == ESP_OK) {
    Serial.printf("[ESP-NOW] Queued filename: %s\n", lastEspNowFilename);
    return true;
  } else {
    Serial.printf("[ESP-NOW] Queue FAILED for '%s' (err=%d)\n", lastEspNowFilename, r);
    return false;
  }
}

// ─────────────────────────────────────────────────────────────
//  Camera
// ─────────────────────────────────────────────────────────────
void initCamera(pixformat_t format, framesize_t size) {
  esp_camera_deinit();
  delay(100);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = format;
  config.frame_size   = size;
  config.jpeg_quality = 10;
  config.fb_count     = 2;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while (true) delay(1000);
  }
}

// ─────────────────────────────────────────────────────────────
//  WiFi
// ─────────────────────────────────────────────────────────────
void connectWiFi() {
  WiFi.mode(WIFI_AP_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.printf("\nConnected! IP: %s  Channel: %d\n",
                WiFi.localIP().toString().c_str(),
                WiFi.channel());
}

void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.println("WiFi lost — reconnecting...");
  WiFi.disconnect();
  delay(300);
  connectWiFi();
}

// ─────────────────────────────────────────────────────────────
//  LED
// ─────────────────────────────────────────────────────────────
void flashConfirmLED() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(400);
    digitalWrite(FLASH_LED_PIN, LOW);
    delay(600);
  }
}

// ─────────────────────────────────────────────────────────────
//  Frame-to-frame motion detection
//  Used only for detecting active motion
// ─────────────────────────────────────────────────────────────
float detectMotionFrameToFrame() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return 0.0f;

  int pixels = fb->width * fb->height;

  if (prevFrame == nullptr || prevFrameSize != pixels) {
    free(prevFrame);
    prevFrame = (uint8_t*)malloc(pixels);
    prevFrameSize = pixels;

    if (!prevFrame) {
      Serial.println("[motion] Failed to allocate prevFrame.");
      esp_camera_fb_return(fb);
      return 0.0f;
    }

    for (int i = 0; i < pixels; i++) {
      prevFrame[i] = rgb565ToLuma(fb->buf, i);
    }

    esp_camera_fb_return(fb);
    return 0.0f;
  }

  int changed = 0;

  for (int i = 0; i < pixels; i++) {
    uint8_t luma = rgb565ToLuma(fb->buf, i);
    if (abs((int)luma - (int)prevFrame[i]) > DIFF_THRESHOLD) {
      changed++;
    }
    prevFrame[i] = luma;
  }

  esp_camera_fb_return(fb);
  return (changed * 100.0f) / pixels;
}

// ─────────────────────────────────────────────────────────────
//  Capture a fixed baseline frame
//  Called once when motion first stops
// ─────────────────────────────────────────────────────────────
bool captureBaselineFrame() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[baseline] Failed to get frame.");
    return false;
  }

  int pixels = fb->width * fb->height;

  if (baselineFrame == nullptr || baselineSize != pixels) {
    free(baselineFrame);
    baselineFrame = (uint8_t*)malloc(pixels);
    baselineSize = pixels;
  }

  if (!baselineFrame) {
    Serial.println("[baseline] Memory allocation failed.");
    esp_camera_fb_return(fb);
    return false;
  }

  for (int i = 0; i < pixels; i++) {
    baselineFrame[i] = rgb565ToLuma(fb->buf, i);
  }

  esp_camera_fb_return(fb);
  Serial.println("[baseline] Fixed baseline captured.");
  return true;
}

// ─────────────────────────────────────────────────────────────
//  Compare current frame against fixed baseline
//  Used during WAITING_FOR_STILL
// ─────────────────────────────────────────────────────────────
float detectMotionAgainstBaseline() {
  if (!baselineFrame) return 0.0f;

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return 0.0f;

  int pixels = fb->width * fb->height;

  if (pixels != baselineSize) {
    Serial.println("[baseline] Size mismatch.");
    esp_camera_fb_return(fb);
    return 0.0f;
  }

  int changed = 0;

  for (int i = 0; i < pixels; i++) {
    uint8_t luma = rgb565ToLuma(fb->buf, i);
    if (abs((int)luma - (int)baselineFrame[i]) > DIFF_THRESHOLD) {
      changed++;
    }
  }

  esp_camera_fb_return(fb);
  return (changed * 100.0f) / pixels;
}

// ─────────────────────────────────────────────────────────────
//  Capture + upload
// ─────────────────────────────────────────────────────────────
bool captureAndSend() {
  initCamera(PIXFORMAT_JPEG, FRAMESIZE_VGA);
  delay(300);

  for (int i = 0; i < 3; i++) {
    camera_fb_t* tmp = esp_camera_fb_get();
    if (tmp) esp_camera_fb_return(tmp);
    delay(50);
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[capture] Capture failed.");
    initCamera(PIXFORMAT_RGB565, FRAMESIZE_QVGA);
    freeMotionBuffers();
    return false;
  }

  Serial.printf("[capture] %u bytes. Uploading...\n", fb->len);

  String boundary = "ESP32boundary";
  String filename = "grocery_" + String(millis()) + "_" + WiFi.macAddress() + ".jpg";
  filename.replace(":", "");

  String head = "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"image\"; filename=\"" + filename + "\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";

  uint32_t totalLen = head.length() + fb->len + tail.length();
  uint8_t* payload  = (uint8_t*)malloc(totalLen);

  if (!payload) {
    Serial.println("[capture] OOM.");
    esp_camera_fb_return(fb);
    initCamera(PIXFORMAT_RGB565, FRAMESIZE_QVGA);
    freeMotionBuffers();
    return false;
  }

  memcpy(payload, head.c_str(), head.length());
  memcpy(payload + head.length(), fb->buf, fb->len);
  memcpy(payload + head.length() + fb->len, tail.c_str(), tail.length());

  esp_camera_fb_return(fb);

  ensureWiFi();

  HTTPClient http;
  String serverUrl = String(SERVER_URL);
  bool isHttps = serverUrl.startsWith("https://");

  WiFiClient client;
  WiFiClientSecure secureClient;
  if (isHttps) {
    secureClient.setInsecure();
    http.begin(secureClient, serverUrl);
  } else {
    http.begin(client, serverUrl);
  }

  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  http.addHeader("Content-Length", String(totalLen));

  Serial.println("[upload] POSTing to server...");
  int code = http.POST(payload, totalLen);
  Serial.printf("[upload] HTTP response code: %d\n", code);

  free(payload);

  bool ok = false;
  bool uploadSuccess = false;

  if (code == 200) {
    String resp = http.getString();
    Serial.println("[upload] ✓ Response: " + resp);
    uploadSuccess = true;
  } else if (code < 0) {
    Serial.printf("[upload] ✗ Connection error: %s\n", http.errorToString(code).c_str());
  } else {
    String resp = http.getString();
    Serial.printf("[upload] ✗ HTTP %d\n", code);
    Serial.println("[upload] Server response:");
    Serial.println(resp);
  }

  Serial.printf("[upload] About to send filename via ESP-NOW: %s\n", filename.c_str());
  bool espNowOK = sendFilenameESPNow(filename);
  Serial.printf("[upload] ESP-NOW queue result: %s\n", espNowOK ? "QUEUED" : "FAILED");

  ok = uploadSuccess;
  
  http.end();
  Serial.println("[upload] http.end() called.");

  initCamera(PIXFORMAT_RGB565, FRAMESIZE_QVGA);
  freeMotionBuffers();

  return ok;
}

// ─────────────────────────────────────────────────────────────
//  setup()
// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(FLASH_LED_PIN, OUTPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);
  digitalWrite(STATUS_LED_PIN, HIGH);

  initCamera(PIXFORMAT_RGB565, FRAMESIZE_QVGA);
  connectWiFi();
  initESPNow();

  Serial.print("Warming up camera");
  for (int i = 0; i < 15; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    Serial.print(".");
    delay(100);
  }
  Serial.println(" done.");
  Serial.println("\n=== Grocery Detection Ready ===");
}

// ─────────────────────────────────────────────────────────────
//  loop()
// ─────────────────────────────────────────────────────────────
void loop() {
  switch (currentState) {
    case IDLE: {
      float changed = detectMotionFrameToFrame();

      if (changed > 0.5f && millis() - lastMotionLogAt >= MOTION_LOG_INTERVAL_MS) {
        Serial.printf("[motion] %.1f%% frame-to-frame | threshold=%.1f%% | state=%s\n",
                      changed, MOTION_PERCENT, stateToString(currentState));
        lastMotionLogAt = millis();
      }

      if (changed >= MOTION_PERCENT) {
        Serial.printf("[→] Motion detected at %.1f%%\n", changed);
        setState(MOTION_DETECTED);
      }
      break;
    }

    case MOTION_DETECTED: {
      float changed = detectMotionFrameToFrame();

      if (changed > 0.5f && millis() - lastMotionLogAt >= MOTION_LOG_INTERVAL_MS) {
        Serial.printf("[motion] %.1f%% frame-to-frame | threshold=%.1f%% | state=%s\n",
                      changed, MOTION_PERCENT, stateToString(currentState));
        lastMotionLogAt = millis();
      }

      if (changed < MOTION_PERCENT) {
        Serial.println("[→] Motion stopped — capturing fixed baseline...");
        if (captureBaselineFrame()) {
          motionStoppedAt = millis();
          setState(WAITING_FOR_STILL);
        } else {
          Serial.println("[baseline] Failed to capture baseline; staying in MOTION_DETECTED.");
        }
      }
      break;
    }

    case WAITING_FOR_STILL: {
      float drift = detectMotionAgainstBaseline();

      if (drift > 0.5f && millis() - lastMotionLogAt >= MOTION_LOG_INTERVAL_MS) {
        Serial.printf("[still-check] %.1f%% changed vs fixed baseline | threshold=%.1f%% | state=%s\n",
                      drift, MOTION_PERCENT, stateToString(currentState));
        lastMotionLogAt = millis();
      }

      if (drift >= MOTION_PERCENT) {
        Serial.printf("[↺] Scene changed again: %.1f%% vs fixed baseline\n", drift);
        setState(MOTION_DETECTED);
        break;
      }

      if (millis() - motionStoppedAt >= MOTION_COOLDOWN_MS) {
        Serial.printf("[→] Scene stable for %d ms (drift=%.1f%%) — capturing now!\n",
                      MOTION_COOLDOWN_MS, drift);
        flashConfirmLED();
        setState(CAPTURING);
      }
      break;
    }

    case CAPTURING: {
      bool ok = captureAndSend();
      Serial.printf("[✓] Capture cycle complete. Upload result: %s\n", ok ? "SUCCESS" : "FAIL");
      Serial.printf("[✓] Locking out for %d s.\n\n", POST_CAPTURE_LOCKOUT_MS / 1000);

      lockoutStart = millis();
      setState(LOCKOUT);

      for (int i = 0; i < 5; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(80);
      }
      break;
    }

    case LOCKOUT:
      if (millis() - lockoutStart >= POST_CAPTURE_LOCKOUT_MS) {
        Serial.println("[→] Lockout expired — back to IDLE.");
        freeMotionBuffers();
        setState(IDLE);
      }
      break;
  }

  delay(100);
}
