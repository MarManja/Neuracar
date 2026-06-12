/**
 * Neuracar — ESP32 SENSORES 
 * ══════════════════════════════════════════════════════════════════
 * Tecnológico de Monterrey, Campus Puebla — MR3002B, 2026
 * Authors: Mariana Manjarrez Lima, Francisco García Uicab
 *
 * Reads AS5047P encoder (SPI init + ABI quadrature runtime) and
 * BNO055 IMU (NDOF fusion mode). Publishes structured telemetry
 * to Jetson Orin Nano via USB-UART at 921600 baud:
 *   E,{angleDeg},{motorRPM},{vLinear}   @ 50 Hz
 *   I,{yaw},{roll},{pitch},{ax},{ay},{az},{gx},{gy},{gz}  @ 50 Hz
 *   STA,{code}  (async, on events)
 *
 * Displays RPM, velocity, angle, IMU orientation, and BNO055
 * calibration status on SSD1306 OLED @ 2 Hz.
 *
 *  PINS
 *  ─────────────────────────────────────────────────────────────────
 *  AS5047P SPI (initial sync):
 *    CS=5   SCK=18   MISO=19   MOSI=23
 *  AS5047P ABI (cuadratura PCNT runtime):
 *    A=32   B=33
 *  OLED SSD1306 128×64: Wire  SDA=21  SCL=22  @0x3C
 *  BNO055:              Wire1 SDA=16  SCL=17  @0x28
 *
 *  PROTOCOL SERIAL @ 921600 baud
 *  ─────────────────────────────────────────────────────────────────
 * ══════════════════════════════════════════════════════════════════
 */

#include <SPI.h>
#include <Wire.h>
#include <ESP32Encoder.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

// ─── PINS ────────────────────────────────────────────────────────
#define CS_PIN      5
#define ENC_A_PIN  32
#define ENC_B_PIN  33

// ─── MECHANICS ───────────────────────────────────────────────────
#define GEAR_RATIO      9.2459f
#define WHEEL_RADIUS_M  0.033f
#define ABI_CPR         4096
#define SPI_RESOLUTION  16384.0f

// ─── HZ ───────────────────────────────────────────────────────────
#define SENSOR_HZ   50
#define OLED_HZ      2

// ─── OLED ─────────────────────────────────────────────────────────
#define SCREEN_W  128
#define SCREEN_H   64
#define OLED_ADDR  0x3C

// ════════════════════════════════════════════════════════════════
//  Objects
// ════════════════════════════════════════════════════════════════

ESP32Encoder     motorEnc;
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);
TwoWire          I2C_IMU = TwoWire(1);
Adafruit_BNO055  bno(55, 0x28, &I2C_IMU);

// ════════════════════════════════════════════════════════════════
//  States
// ════════════════════════════════════════════════════════════════

portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;

bool     g_imuOk    = false;
bool     g_oledOk   = false;
bool     g_shutdown = false;

uint32_t g_lastSensorMs = 0;

// Encoder
int64_t  g_encLastCount = 0;
uint32_t g_encLastUs    = 0;
volatile float g_motorRPM = 0.0f;
volatile float g_vLinear  = 0.0f;
volatile float g_angleDeg = 0.0f;

// IMU
volatile float g_yaw=0, g_roll=0, g_pitch=0;
volatile uint8_t g_calSys=0, g_calGyro=0, g_calAccel=0, g_calMag=0;

// TX buffer
static char g_txBuf[160];
#define SERIAL_SEND(fmt, ...) \
    do { snprintf(g_txBuf, sizeof(g_txBuf), fmt, ##__VA_ARGS__); \
         Serial.write((const uint8_t*)g_txBuf, strlen(g_txBuf)); } while(0)

// ════════════════════════════════════════════════════════════════
//  AS5047P — SPI initial sync 
// ════════════════════════════════════════════════════════════════

static uint16_t spi16(uint16_t tx) {
    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE1));
    digitalWrite(CS_PIN, LOW); delayMicroseconds(1);
    uint16_t hi = SPI.transfer((tx >> 8) & 0xFF);
    uint16_t lo = SPI.transfer(tx & 0xFF);
    delayMicroseconds(1); digitalWrite(CS_PIN, HIGH);
    SPI.endTransaction(); delayMicroseconds(2);
    return (hi << 8) | lo;
}
static uint16_t addParity(uint16_t c) {
    uint16_t p = c & 0x7FFF;
    p ^= p>>8; p ^= p>>4; p ^= p>>2; p ^= p>>1;
    if (p & 1) c |= 0x8000; return c;
}
static uint16_t spiReadAngle() {
    spi16(addParity(0x7FFF));
    return spi16(addParity(0x4000)) & 0x3FFF;
}
static void syncEncoderToAbsolute() {
    uint16_t raw     = spiReadAngle();
    int64_t  initCnt = (int64_t)((float)raw / SPI_RESOLUTION * ABI_CPR);
    motorEnc.setCount(initCnt);
    g_encLastCount = initCnt;
    g_encLastUs    = micros();
    g_angleDeg     = (float)initCnt / ABI_CPR * 360.0f;
    SERIAL_SEND("STA,ENC_SYNC,%.2f\n", raw * 360.0f / SPI_RESOLUTION);
}

// ════════════════════════════════════════════════════════════════
//  Encoder 
// ════════════════════════════════════════════════════════════════

static void readEncoder() {
    int64_t  count = motorEnc.getCount();
    uint32_t nowUs = micros();
    uint32_t dtUs  = nowUs - g_encLastUs;
    if (dtUs < 100) return;
    int64_t delta = count - g_encLastCount;
    float   dtS   = dtUs * 1e-6f;
    float omegaMotor = ((float)delta / ABI_CPR) * TWO_PI / dtS;
    portENTER_CRITICAL(&g_mux);
    g_vLinear  = (omegaMotor / GEAR_RATIO) * WHEEL_RADIUS_M;
    g_motorRPM = omegaMotor * 60.0f / TWO_PI;
    g_angleDeg = (float)count / ABI_CPR * 360.0f;
    portEXIT_CRITICAL(&g_mux);
    g_encLastCount = count;
    g_encLastUs    = nowUs;
}

// ════════════════════════════════════════════════════════════════
//  IMU 
// ════════════════════════════════════════════════════════════════

static void initIMU() {
    g_imuOk = bno.begin();
    if (!g_imuOk) { Serial.write("STA,ERR_IMU\n", 12); return; }
    bno.setExtCrystalUse(true);
    bno.setAxisRemap(Adafruit_BNO055::REMAP_CONFIG_P1);
    bno.setAxisSign(Adafruit_BNO055::REMAP_SIGN_P1);
    delay(50);
    Serial.write("STA,IMU_OK\n", 11);
}

static void readIMU() {
    if (!g_imuOk) return;
    sensors_event_t evOri, evAccel, evGyro;
    bno.getEvent(&evOri,   Adafruit_BNO055::VECTOR_EULER);
    bno.getEvent(&evAccel, Adafruit_BNO055::VECTOR_LINEARACCEL);
    bno.getEvent(&evGyro,  Adafruit_BNO055::VECTOR_GYROSCOPE);
    uint8_t cs=0,cg=0,ca=0,cm=0;
    bno.getCalibration(&cs,&cg,&ca,&cm);
    portENTER_CRITICAL(&g_mux);
    g_yaw=evOri.orientation.x; g_roll=evOri.orientation.y; g_pitch=evOri.orientation.z;
    g_calSys=cs; g_calGyro=cg; g_calAccel=ca; g_calMag=cm;
    portEXIT_CRITICAL(&g_mux);
    SERIAL_SEND("I,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        evOri.orientation.x, evOri.orientation.y, evOri.orientation.z,
        evAccel.acceleration.x, evAccel.acceleration.y, evAccel.acceleration.z,
        evGyro.gyro.x, evGyro.gyro.y, evGyro.gyro.z);
}

// ════════════════════════════════════════════════════════════════
//  OLED 
//
//  128×64 px, text size 1 (6×8 px/char), refresh rate @ 2 Hz
//
//  Y=0  "Neuracar  SENSORES" 
//  Y=14  RPM motor + wheel linear speed
//  Y=26  Encoder absolute angle
//  Y=38  Yaw / Roll / Pitch  (o "IMU: ERROR")
//  Y=52  BNO055  S/G/A/M  (0–3) calibration
//        If SHUTDOWN → "    APAGANDO...    "
// ════════════════════════════════════════════════════════════════

static void oledTask(void* pv) {
    const TickType_t xDelay = pdMS_TO_TICKS(1000 / OLED_HZ);
    char buf[22];
    for (;;) {
        vTaskDelay(xDelay);
        if (!g_oledOk) continue;

        portENTER_CRITICAL(&g_mux);
        float rpm=g_motorRPM, vl=g_vLinear, ang=g_angleDeg;
        float yaw=g_yaw, roll=g_roll, pitch=g_pitch;
        bool imuOk=g_imuOk, sd=g_shutdown;
        uint8_t cs=g_calSys, cg=g_calGyro, ca=g_calAccel, cm=g_calMag;
        portEXIT_CRITICAL(&g_mux);

        display.clearDisplay();
        display.setTextColor(SSD1306_WHITE);
        display.setTextSize(1);

        display.setCursor(0, 0);
        display.print("Neuracar  SENSORES");
        display.drawLine(0, 9, 127, 9, SSD1306_WHITE);

        display.setCursor(0, 14);
        snprintf(buf, sizeof(buf), "RPM:%+.0f V:%+.2fm/s", rpm, vl);
        display.print(buf);

        display.setCursor(0, 26);
        snprintf(buf, sizeof(buf), "Ang: %.1f deg", ang);
        display.print(buf);

        display.setCursor(0, 38);
        if (imuOk) {
            snprintf(buf, sizeof(buf), "Y:%3.0f R:%+.0f P:%+.0f", yaw, roll, pitch);
            display.print(buf);
        } else { display.print("IMU: ERROR"); }

        display.setCursor(0, 52);
        if (sd) {
            display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
            display.print("    APAGANDO...     ");
            display.setTextColor(SSD1306_WHITE);
        } else {
            snprintf(buf, sizeof(buf), "CAL: S%d G%d A%d M%d", cs, cg, ca, cm);
            display.print(buf);
        }

        display.display();   
    }
}

// ════════════════════════════════════════════════════════════════
//  Serial Parser
// ════════════════════════════════════════════════════════════════

static char    g_rxBuf[32];
static uint8_t g_rxIdx = 0;

static void readSerial() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            g_rxBuf[g_rxIdx] = '\0';
            if (strncmp(g_rxBuf, "SHUTDOWN", 8) == 0) {
                portENTER_CRITICAL(&g_mux);
                g_shutdown = true;
                portEXIT_CRITICAL(&g_mux);
                Serial.write("STA,SHUTDOWN_ACK\n", 17);
            }
            g_rxIdx = 0;
        } else if (c != '\r' && g_rxIdx < sizeof(g_rxBuf)-1) {
            g_rxBuf[g_rxIdx++] = c;
        } else if (g_rxIdx >= sizeof(g_rxBuf)-1) {
            g_rxIdx = 0;
        }
    }
}

// ════════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════════

void setup() {
    Serial.setTxBufferSize(4096);
    Serial.setRxBufferSize(512);
    Serial.begin(921600);
    delay(100);

    pinMode(CS_PIN, OUTPUT);
    digitalWrite(CS_PIN, HIGH);
    SPI.begin(18, 19, 23, CS_PIN);
    delay(100);

    Wire.begin(21, 22);    Wire.setClock(400000);    
    I2C_IMU.begin(16, 17); I2C_IMU.setClock(400000); 

    g_oledOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
    if (g_oledOk) {
        display.clearDisplay(); display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0,0);  display.print("Neuracar  SENSORES");
        display.drawLine(0,9,127,9,SSD1306_WHITE);
        display.setCursor(0,14); display.print("Iniciando...");
        display.display();
    } else {
        Serial.write("STA,WARN_OLED_NOT_FOUND\n", 24);
    }

    initIMU();

    ESP32Encoder::useInternalWeakPullResistors = puType::up;
    motorEnc.attachFullQuad(ENC_A_PIN, ENC_B_PIN);
    motorEnc.setCount(0);
    delay(50);
    syncEncoderToAbsolute();

    xTaskCreatePinnedToCore(oledTask, "OLED", 4096, NULL, 1, NULL, 0);

    g_lastSensorMs = millis();

    Serial.write("STA,READY_SENSORES\n", 19);
}

// ════════════════════════════════════════════════════════════════
//  LOOP 
// ════════════════════════════════════════════════════════════════

void loop() {
    uint32_t now = millis();

    readSerial();

    if (g_shutdown) return;

    if (now - g_lastSensorMs >= (1000U / SENSOR_HZ)) {
        g_lastSensorMs = now;
        readEncoder();
        SERIAL_SEND("E,%.2f,%.2f,%.4f\n", g_angleDeg, g_motorRPM, g_vLinear);
        readIMU();
    }
}
