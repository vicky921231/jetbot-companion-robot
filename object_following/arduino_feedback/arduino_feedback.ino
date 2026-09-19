/*
 * arduino_feedback —— JetBot 伴護機器人的紅綠燈 / 蜂鳴器 / 雷射測距韌體
 *
 * 2026-09-18 改版。相對前一版的三個變動:
 *   1. 單顆 LED → 交通號誌模組(紅/黃/綠三色共陰)
 *   2. 新增 VL53L0X 雷射測距(I2C),以 10Hz **主動推播**距離
 *   3. 鮑率 9600 → 115200
 *
 * ⚠️ 鮑率變了,組員既有的測試腳本要跟著改一個數字。
 *    非改不可的理由:9600 下「問一句答一句」拿一次距離要約 17ms,
 *    而 15 FPS 的每幀預算只有 66ms,光要個距離就吃掉四分之一。
 *    改成 115200 + 主動推播 + Jetson 端背景執行緒讀取後,callback 完全不必等。
 *
 * LED_ON / LED_OFF 指令保留(映射到紅燈),舊的測試腳本仍然能用。
 *
 * ---------------------------------------------------------------------------
 * 接線
 * ---------------------------------------------------------------------------
 *
 * 交通號誌模組(共陰,4 pin):
 *     GND → Arduino GND
 *     R   → D10   (沿用前一版已實測的腳位,高電位亮)
 *     Y   → D9
 *     G   → D8
 *
 *   ⚠️ 先確認模組上有沒有限流電阻。看不出來就每一路串 220Ω ——
 *      直接接會超過 Arduino 單腳 20mA 上限,久了會燒腳位。
 *      串了只是稍微暗一點,沒有壞處。
 *
 * 有源蜂鳴器:
 *     +   → D11
 *     -   → GND
 *
 * VL53L0X 雷射測距(GY-530):
 *     VIN → 5V     (板上有穩壓,3~5V 都可以)
 *     GND → GND
 *     SCL → A5     ← Uno 的 I2C 腳位固定,不能改
 *     SDA → A4
 *
 *   取代原本的 HC-SR04。換掉之後「ECHO 輸出 5V 會燒 Jetson」那個地雷消失,
 *   因為測距現在完全在 Arduino 這邊,而且走 I2C 不是自己掐秒錶。
 *
 * ---------------------------------------------------------------------------
 * 通訊協定(115200 8N1)
 * ---------------------------------------------------------------------------
 *
 * Jetson → Arduino:
 *     LIGHT:RYG    三顆一次設定,R/Y/G 各為 0 或 1。例:LIGHT:100 = 只亮紅燈
 *     LED_ON       等同 LIGHT:100(向後相容)
 *     LED_OFF      等同 LIGHT:000(向後相容)
 *     BEEP         蜂鳴器短鳴 150ms
 *     BEEP:<ms>    指定長度,上限 3000
 *     PING         連線檢查
 *     DIST         單次查詢距離(除錯用;正常運作靠下面的主動推播)
 *
 * Arduino → Jetson:
 *     READY vl53l0x=ok|absent    開機訊息
 *     D:<mm>                     距離,約每 100ms 一筆。-1 = 超出量程或讀取失敗
 *     LIGHT <ryg> / LED ON / LED OFF / BEEP / PONG / ERR <原因>
 *
 * ---------------------------------------------------------------------------
 * 為什麼全部用 millis() 不用 delay()
 * ---------------------------------------------------------------------------
 *
 * delay() 期間收不到序列埠指令,紅綠燈會卡住不閃。
 * 同理,測距用**連續模式**而不是每次呼叫 rangingTest():
 * 後者會阻塞約 30ms,那段時間 Arduino 的 64 bytes 接收緩衝區可能被塞爆。
 *
 * 對應 feedback.py 的 ArduinoBackend 與 rangefinder.py。
 */

#include <Wire.h>
#include <Adafruit_VL53L0X.h>

// --- 腳位 ------------------------------------------------------------------

const int PIN_RED    = 10;   // 已實測:高電位亮
const int PIN_YELLOW = 9;
const int PIN_GREEN  = 8;
const int PIN_BUZZER = 11;

// 有源蜂鳴器:給高電位就會叫,不需要 PWM。
// 若買到的是無源型,把 buzzerOn() / buzzerOff() 換成 tone() / noTone()。
const bool ACTIVE_BUZZER = true;

// --- 參數 ------------------------------------------------------------------

const unsigned long DEFAULT_BEEP_MS = 150;
const unsigned long MAX_BEEP_MS = 3000;      // 上限,避免打錯字變成長鳴不停

const unsigned long RANGE_PERIOD_MS = 100;   // 連續測距週期 = 10Hz
const unsigned long RANGE_POLL_MS = 20;      // 多久去問一次「量好了沒」
const uint16_t MAX_RANGE_MM = 2000;          // 超過此距離的讀數不可靠,一律回 -1

// --- 狀態 ------------------------------------------------------------------

Adafruit_VL53L0X lox;
bool rangeReady = false;              // false = 感測器沒接或初始化失敗
unsigned long lastRangePoll = 0;

String inputString = "";
unsigned long beepUntil = 0;          // 0 代表沒在響


// --- 輸出 ------------------------------------------------------------------

void setLights(bool red, bool yellow, bool green) {
  digitalWrite(PIN_RED,    red    ? HIGH : LOW);
  digitalWrite(PIN_YELLOW, yellow ? HIGH : LOW);
  digitalWrite(PIN_GREEN,  green  ? HIGH : LOW);
}

void buzzerOn() {
  if (ACTIVE_BUZZER) {
    digitalWrite(PIN_BUZZER, HIGH);
  } else {
    tone(PIN_BUZZER, 2000);
  }
}

void buzzerOff() {
  if (ACTIVE_BUZZER) {
    digitalWrite(PIN_BUZZER, LOW);
  } else {
    noTone(PIN_BUZZER);
  }
}

void startBeep(unsigned long ms) {
  if (ms == 0 || ms > MAX_BEEP_MS) {
    Serial.println("ERR beep length");
    return;
  }
  buzzerOn();
  beepUntil = millis() + ms;
  Serial.println("BEEP");
}


// --- 指令 ------------------------------------------------------------------

void handleLight(const String& arg) {
  // 期待剛好三個字元,每個是 '0' 或 '1'
  if (arg.length() != 3) {
    Serial.println("ERR light format");
    return;
  }
  for (int i = 0; i < 3; i++) {
    if (arg[i] != '0' && arg[i] != '1') {
      Serial.println("ERR light format");
      return;
    }
  }
  setLights(arg[0] == '1', arg[1] == '1', arg[2] == '1');
  Serial.print("LIGHT ");
  Serial.println(arg);
}

void reportRangeOnce() {
  if (!rangeReady) {
    Serial.println("D:-1");
    return;
  }
  uint16_t mm = lox.readRange();
  if (lox.readRangeStatus() == 4 || mm == 0 || mm > MAX_RANGE_MM) {
    Serial.println("D:-1");
  } else {
    Serial.print("D:");
    Serial.println(mm);
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) {
    return;
  }

  if (cmd.startsWith("LIGHT:")) {
    handleLight(cmd.substring(6));

  } else if (cmd == "LED_ON") {          // 向後相容:映射到紅燈
    setLights(true, false, false);
    Serial.println("LED ON");

  } else if (cmd == "LED_OFF") {         // 向後相容:全滅
    setLights(false, false, false);
    Serial.println("LED OFF");

  } else if (cmd == "BEEP") {
    startBeep(DEFAULT_BEEP_MS);

  } else if (cmd.startsWith("BEEP:")) {
    startBeep((unsigned long) cmd.substring(5).toInt());

  } else if (cmd == "PING") {
    Serial.print("PONG vl53l0x=");
    Serial.println(rangeReady ? "ok" : "absent");

  } else if (cmd == "DIST") {
    reportRangeOnce();

  } else {
    Serial.print("ERR unknown ");
    Serial.println(cmd);
  }
}


// --- 主迴圈的三個工作 -------------------------------------------------------

void serviceSerial() {
  while (Serial.available() > 0) {
    char c = (char) Serial.read();
    if (c == '\n') {
      handleCommand(inputString);
      inputString = "";
    } else if (c != '\r') {
      // 防呆:亂碼或沒有換行的情況下不要讓字串無限長大
      if (inputString.length() < 40) {
        inputString += c;
      }
    }
  }
}

void serviceBeep() {
  // 用差值比大小而不是 millis() >= beepUntil:millis() 約 50 天會溢位歸零,
  // 直接比大小會在溢位那一刻讓蜂鳴器停不下來。
  if (beepUntil != 0 && (long)(millis() - beepUntil) >= 0) {
    buzzerOff();
    beepUntil = 0;
  }
}

void serviceRange() {
  if (!rangeReady) {
    return;
  }
  // isRangeComplete() 每次呼叫都要走一趟 I2C。主迴圈一秒跑幾萬圈,
  // 不節流會把 I2C 匯流排佔滿,也會拖慢序列埠處理。
  unsigned long now = millis();
  if (now - lastRangePoll < RANGE_POLL_MS) {
    return;
  }
  lastRangePoll = now;

  if (lox.isRangeComplete()) {
    reportRangeOnce();
  }
}


// --- setup / loop ----------------------------------------------------------

void setup() {
  Serial.begin(115200);
  inputString.reserve(48);

  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_YELLOW, OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  setLights(false, false, false);
  buzzerOff();

  // 感測器還沒接線也要能正常開機,只是不推播距離。
  // 這樣可以先燒錄韌體、先把紅綠燈測完,之後再接 VL53L0X。
  if (lox.begin()) {
    lox.startRangeContinuous(RANGE_PERIOD_MS);
    rangeReady = true;
  }

  Serial.print("READY vl53l0x=");
  Serial.println(rangeReady ? "ok" : "absent");
}

void loop() {
  serviceSerial();
  serviceBeep();
  serviceRange();
}

/*
 * ---------------------------------------------------------------------------
 * 如果編譯不過 / 行為怪怪的
 * ---------------------------------------------------------------------------
 *
 * 1. 找不到 Adafruit_VL53L0X.h
 *    Arduino IDE → 工具 → 管理程式庫 → 搜尋 "Adafruit VL53L0X" → 安裝
 *    (它會一併要求安裝 Adafruit BusIO,按同意)
 *
 * 2. startRangeContinuous / isRangeComplete / readRange 未定義
 *    程式庫版本太舊。升級到 1.2.0 以上。
 *    真的升不了的話,把 serviceRange() 換成下面這段阻塞版本 ——
 *    可以動,但每次量測會卡住迴圈約 30ms:
 *
 *      void serviceRange() {
 *        if (!rangeReady) return;
 *        unsigned long now = millis();
 *        if (now - lastRangePoll < RANGE_PERIOD_MS) return;
 *        lastRangePoll = now;
 *        VL53L0X_RangingMeasurementData_t measure;
 *        lox.rangingTest(&measure, false);
 *        if (measure.RangeStatus != 4 && measure.RangeMilliMeter <= MAX_RANGE_MM) {
 *          Serial.print("D:"); Serial.println(measure.RangeMilliMeter);
 *        } else {
 *          Serial.println("D:-1");
 *        }
 *      }
 *
 *    並把 setup() 裡的 startRangeContinuous() 那一行刪掉。
 *
 * 3. 燒錄後序列埠監控視窗全是亂碼
 *    鮑率沒改。右下角選 115200。
 *
 * 4. 開機顯示 vl53l0x=absent
 *    感測器沒接、接錯腳位、或 VIN 沒供電。
 *    這不會讓其他功能失效 —— 紅綠燈和蜂鳴器照常運作,只是不推播距離。
 */
