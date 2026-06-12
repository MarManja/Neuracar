/**
 * Neuracar — ESP32 ACTUADORES  
 * ══════════════════════════════════════════════════════════════════
 * Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026
 * Authors: Mariana Manjarrez Lima, Maximiliano Cruz Rascón
 *
 * Receives C,{throttle},{steering} frames from Jetson via
 * USB-UART at 921600 baud and generates 50 Hz PWM signals
 * for Traxxas XL-5 ESC and steering servo.
 * Implements latest-only TX pattern and 500 ms watchdog.
 *
 *  PINS
 *  ─────────────────────────────────────────────────────────────────
 *  ESC signal:    GPIO 25
 *  Servo signal:  GPIO 26
 *  Common GND with ESP32-S, ESC and Servo
 *
 *  PROTOCOL SERIAL @ 921600 baud
 *  ─────────────────────────────────────────────────────────────────
 * ══════════════════════════════════════════════════════════════════
 */

#include <ESP32Servo.h>

// ─── PINES ────────────────────────────────────────────────────────
#define ESC_PIN    25
#define SERVO_PIN  26

// ─── ESC / SERVO   ────────────────────────────────────────────────
#define ESC_NEUTRAL   1512
#define ESC_FORWARD   1750
#define ESC_REVERSE   1274
#define SERVO_CENTER  1549
#define SERVO_LEFT    1101 // 448
#define SERVO_RIGHT   1997 // 448

// ════════════════════════════════════════════════════════════════
//  Objects
// ════════════════════════════════════════════════════════════════

Servo esc;
Servo steeringServo;

// ════════════════════════════════════════════════════════════════
//  States
// ════════════════════════════════════════════════════════════════

float g_throttle = 0.0f;
float g_steering = 0.0f;
bool  g_estop    = false;
bool  g_shutdown = false;

static char g_txBuf[64];
#define SERIAL_SEND(fmt, ...) \
    do { snprintf(g_txBuf, sizeof(g_txBuf), fmt, ##__VA_ARGS__); \
         Serial.write((const uint8_t*)g_txBuf, strlen(g_txBuf)); } while(0)

// ════════════════════════════════════════════════════════════════
//  Actuators
// ════════════════════════════════════════════════════════════════

static int normalizedToPwm(float v, int pwmMin, int pwmNeutral, int pwmMax) {
    v = constrain(v, -1.0f, 1.0f);
    return (v >= 0.0f)
        ? pwmNeutral + (int)((float)(pwmMax - pwmNeutral) * v)
        : pwmNeutral + (int)((float)(pwmNeutral - pwmMin) * v);
}

static void neutralActuators() {
    esc.writeMicroseconds(ESC_NEUTRAL);
    steeringServo.writeMicroseconds(SERVO_CENTER);
    g_throttle = 0.0f;
    g_steering = 0.0f;
}

static void applyCommands(float throttle, float steering) {
    if (g_estop || g_shutdown) return;
    esc.writeMicroseconds(constrain(
        normalizedToPwm(throttle, ESC_REVERSE, ESC_NEUTRAL, ESC_FORWARD),
        ESC_REVERSE, ESC_FORWARD));
    steeringServo.writeMicroseconds(constrain(
        normalizedToPwm(steering, SERVO_LEFT, SERVO_CENTER, SERVO_RIGHT),
        SERVO_LEFT, SERVO_RIGHT));
}

// ════════════════════════════════════════════════════════════════
//  Serial Parser
// ════════════════════════════════════════════════════════════════

static char    g_rxBuf[64];
static uint8_t g_rxIdx = 0;

static void parseLine(char* line) {
    if (line[0] == 'C' && line[1] == ',') {
        float t = 0.0f, s = 0.0f;
        if (sscanf(line + 2, "%f,%f", &t, &s) == 2) {
            g_throttle = constrain(t, -1.0f, 1.0f);
            g_steering = constrain(s, -1.0f, 1.0f);
            applyCommands(g_throttle, g_steering);
        }
        return;
    }

    if (strncmp(line, "SHUTDOWN", 8) == 0) {
        g_shutdown = true;
        neutralActuators();
        Serial.write("STA,SHUTDOWN_ACK\n", 17);
        return;
    }

    if (strncmp(line, "CLRESTOP", 8) == 0) {
        g_estop = false;
        Serial.write("STA,ESTOP_CLEAR\n", 16);
        return;
    }

}

static void readSerial() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            g_rxBuf[g_rxIdx] = '\0';
            if (g_rxIdx > 0) parseLine(g_rxBuf);
            g_rxIdx = 0;
        } else if (c != '\r' && g_rxIdx < sizeof(g_rxBuf) - 1) {
            g_rxBuf[g_rxIdx++] = c;
        } else if (g_rxIdx >= sizeof(g_rxBuf) - 1) {
            g_rxIdx = 0; 
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════════

void setup() {
    Serial.setRxBufferSize(2048);
    Serial.setTxBufferSize(256);
    Serial.begin(921600);
    delay(100);

    esc.attach(ESC_PIN, 1000, 2000);
    steeringServo.attach(SERVO_PIN, 1000, 2000);
    neutralActuators();

    delay(2000);

    Serial.write("STA,READY_ACTUADORES\n", 21);
}

// ════════════════════════════════════════════════════════════════
//  LOOP
// ════════════════════════════════════════════════════════════════

void loop() {
    readSerial();
}
