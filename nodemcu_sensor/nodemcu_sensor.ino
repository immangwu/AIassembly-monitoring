/*
 * TANCAM Valve Assembly — NodeMCU Ultrasonic Sensor Server
 * Board  : NodeMCU 1.0 (ESP8266)
 * Sensor : HC-SR04 Ultrasonic Distance Sensor
 *
 * Wiring:
 *   HC-SR04 VCC  → NodeMCU 3.3V (or Vin for 5V)
 *   HC-SR04 GND  → NodeMCU GND
 *   HC-SR04 TRIG → NodeMCU D6 (GPIO12)
 *   HC-SR04 ECHO → NodeMCU D7 (GPIO13)
 *
 * Once running, open Serial Monitor at 115200 baud to get the IP address.
 * Then enter that IP in the Streamlit app Stage 5 field.
 *
 * Endpoint:  GET http://<NodeMCU-IP>/distance
 * Response:  {"distance": 8.4, "unit": "cm", "status": "ok"}
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// ── WiFi credentials ─────────────────────────────────────
const char* SSID     = "YOUR_WIFI_SSID";      // <-- change this
const char* PASSWORD = "YOUR_WIFI_PASSWORD";   // <-- change this

// ── HC-SR04 pins ─────────────────────────────────────────
const int TRIG_PIN = D6;   // GPIO12
const int ECHO_PIN = D7;   // GPIO13

// ── LED (built-in, active LOW) ───────────────────────────
const int LED_PIN  = LED_BUILTIN;

ESP8266WebServer server(80);

// ─────────────────────────────────────────────────────────
// Measure distance in cm (returns -1.0 on timeout)
// Positioning Length = how close the assembly is to the sensor
// ─────────────────────────────────────────────────────────
float measureDistance() {
    // Send 10µs trigger pulse
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(4);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    // Measure echo pulse (timeout = 30 ms → max ~5 m range)
    long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);

    if (duration == 0) return -1.0;  // no echo / out of range

    // Distance = (duration × speed-of-sound) / 2
    // Speed of sound ≈ 0.0343 cm/µs
    return (duration * 0.0343f) / 2.0f;
}

// ─────────────────────────────────────────────────────────
// HTTP handler  GET /distance
// ─────────────────────────────────────────────────────────
void handleDistance() {
    // Allow cross-origin requests from Streamlit
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Cache-Control", "no-cache");

    float dist = measureDistance();

    String json;
    if (dist < 0) {
        json = "{\"distance\": -1, \"unit\": \"cm\", \"status\": \"timeout\"}";
    } else {
        json  = "{\"distance\": ";
        json += String(dist, 1);
        json += ", \"unit\": \"cm\", \"status\": \"ok\"}";
    }

    // Blink LED to confirm request received
    digitalWrite(LED_PIN, LOW);
    delay(30);
    digitalWrite(LED_PIN, HIGH);

    unsigned long t_ms = millis();
    server.send(200, "application/json", json);
    Serial.printf("[REQ] /distance → %.1f cm  (resp ~%lu ms)\n", dist, millis() - t_ms);
}

// ─────────────────────────────────────────────────────────
// HTTP handler  GET /torque
// Connect a flex sensor or FSR (Force Sensitive Resistor) to A0
// Returns estimated torque in Nm (scale factor may need calibration)
//
// Wiring (optional torque sensor):
//   FSR / Flex-sensor between A0 and GND, 10kΩ pull-up to 3.3V
// ─────────────────────────────────────────────────────────
void handleTorque() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Cache-Control", "no-cache");

    // Read analog value (0–1023 on ESP8266 A0, input range 0–1.0V)
    int raw = analogRead(A0);

    // Linear scale: 0 raw = 0 Nm, 1023 raw = 20 Nm (calibrate to your sensor)
    float torque_nm = (raw / 1023.0f) * 20.0f;

    String json  = "{\"torque\": ";
    json += String(torque_nm, 2);
    json += ", \"raw\": ";
    json += String(raw);
    json += ", \"unit\": \"Nm\", \"target\": 15.0, \"status\": \"";
    json += (torque_nm >= 13.0 && torque_nm <= 17.0) ? "ok" :
            (torque_nm > 17.0)  ? "over" : "under";
    json += "\"}";

    digitalWrite(LED_PIN, LOW); delay(30); digitalWrite(LED_PIN, HIGH);
    server.send(200, "application/json", json);
    Serial.printf("[REQ] /torque → %.2f Nm  (raw=%d)\n", torque_nm, raw);
}

// ─────────────────────────────────────────────────────────
// HTTP handler  GET /  (root info page)
// ─────────────────────────────────────────────────────────
void handleRoot() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    String html = "<h2>TANCAM Sensor Node</h2>"
                  "<p><a href='/distance'>/distance</a> — HC-SR04 positioning length (cm)</p>"
                  "<p><a href='/torque'>/torque</a> — FSR torque sensor (Nm)</p>";
    server.send(200, "text/html", html);
}

void handleNotFound() {
    server.send(404, "text/plain", "Not found");
}

// ─────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(100);

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);   // LED off (active LOW)
    digitalWrite(TRIG_PIN, LOW);

    Serial.println("\n\n=== TANCAM Sensor Node ===");
    Serial.printf("Connecting to WiFi: %s\n", SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(SSID, PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        attempts++;
        if (attempts > 40) {
            Serial.println("\n[ERROR] WiFi failed. Restarting...");
            ESP.restart();
        }
    }

    Serial.println("\n[OK] WiFi connected!");
    Serial.print("[IP] http://");
    Serial.println(WiFi.localIP());
    Serial.println("[URL] http://" + WiFi.localIP().toString() + "/distance");
    Serial.println("Enter the IP above into the Streamlit app Stage 5 field.");

    server.on("/",         handleRoot);
    server.on("/distance", handleDistance);
    server.on("/torque",   handleTorque);
    server.onNotFound(handleNotFound);
    server.begin();

    Serial.println("[SERVER] HTTP server started on port 80");

    // Blink 3x to signal ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, LOW);  delay(150);
        digitalWrite(LED_PIN, HIGH); delay(150);
    }
}

void loop() {
    server.handleClient();
}
