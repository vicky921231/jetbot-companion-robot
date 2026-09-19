"""目標者的即時回饋(紅綠燈 + 蜂鳴器)、序列埠連線、與照護者的遠端通知。

設計原則(需求文件「LINE Notify 的定位」章節):
  - 目標者本人的即時回饋 → 完全由紅綠燈 + 蜂鳴器負責
  - 照護者的遠端狀態紀錄 → 走 notify_caregiver(),與燈號平行、不互相取代
  - 蜂鳴器只在「鎖定完成」那一刻使用;跟丟、無法跨越障礙一律不發警示音

沒有 LCD,不需要 I2C 文字驅動。

2026-09-18 改版:
  - 單顆 LED → **交通號誌模組**(紅/黃/綠)。順帶解決一個缺陷:
    單顆燈時 waiting 與 impassable_wait 都是慢閃、長得一模一樣,
    但這兩個狀態意義天差地遠(一個是「等你站好」、一個是「我卡住了」)。
  - ArduinoBackend 改成 **115200 + 背景讀取執行緒**,並接手 VL53L0X 推播的距離。
    序列埠只有一個擁有者(這支檔案),rangefinder.py 向它要數字,不自己開連線。

硬體後端有三種,由 BACKEND 決定:
  'arduino'  接在 Arduino 上,Jetson 透過 USB 序列埠下指令(目前採用)
  'gpio'     直接接 Jetson Nano 的 40-pin GPIO(備案,沒有測距功能)
  'print'    沒有硬體,只印字。筆電上會自動退到這個模式

不論哪一種,對外介面都一樣(update_lights / beep_locked / close),
所以 notebook 與狀態機完全不需要知道硬體接在哪裡。
"""

import threading
import time
from collections import deque

from follow_state import (
    STATE_WAITING, STATE_CONFIRMING, STATE_LOCKED, STATE_DETOUR, STATE_IMPASSABLE,
)

# --- 後端選擇 -------------------------------------------------------------
BACKEND = 'arduino'

# --- Arduino 後端設定 -----------------------------------------------------
# TODO: 確認實際的序列埠。Arduino Uno/Nano 在 Jetson 上通常是 /dev/ttyACM0,
#       用 CH340 晶片的副廠板子則是 /dev/ttyUSB0。
#       在 JetBot 的 bash 執行 `ls /dev/ttyACM* /dev/ttyUSB*` 可以確認。
ARDUINO_PORT = '/dev/ttyACM0'

# ⚠️ 2026-09-18 從 9600 改成 115200,**必須與 arduino_feedback.ino 一致**。
#    組員既有的測試腳本也要跟著改這個數字。
#    非改不可的理由:9600 下拿一次距離要約 17ms,而 15 FPS 的每幀預算只有 66ms。
ARDUINO_BAUD = 115200
ARDUINO_RESET_WAIT = 2.0   # 開啟序列埠會讓 Arduino 重開機,要等它啟動完

# --- Jetson GPIO 後端設定(BACKEND = 'gpio' 時才用到)---------------------
# BOARD 編號。原本 sonar.py 佔用的 16/22 已經釋出(測距改到 Arduino)。
RED_PIN = 15
YELLOW_PIN = 16
GREEN_PIN = 18
BUZZER_PIN = 22

# --- 共用 -----------------------------------------------------------------
BEEP_SECONDS = 0.15        # 鎖定完成的短鳴長度
SLOW_PERIOD = 1.0          # 慢速閃爍週期(秒)
FAST_PERIOD = 0.2          # 快速閃爍週期(秒)

# 燈的極性:True = 高電位亮(紅燈已於 2026-09-13 實測確認;號誌模組為共陰,同樣邏輯)
ACTIVE_HIGH = True

# 距離讀數多久沒更新就視為失效(秒)。Arduino 以 10Hz 推播,0.5 秒等於漏了 5 筆。
DISTANCE_STALE_SECONDS = 0.5
DISTANCE_WINDOW = 5        # 中位數濾波的樣本數


# --- 狀態 → 燈號 -----------------------------------------------------------
#
# 三色之後才做得到的事:waiting(紅慢閃)與 impassable_wait(紅恆亮)終於分得開。
#
# detour 刻意維持「綠燈恆亮」與 locked 相同 —— 繞行期間鎖定並沒有解除,
# 對目標者而言「還跟著你」的訊息不該改變。黃燈是額外疊上去的資訊,
# 不推翻原本的設計決定。
#
# 交通號誌的語意對長輩是零學習成本:綠=跟著走、黃=等一下、紅=停。

_SOLID = 'solid'
_SLOW = 'slow'
_FAST = 'fast'
_OFF = 'off'

STATE_LIGHTS = {
    #                    紅      黃      綠
    STATE_WAITING:    (_SLOW,  _OFF,   _OFF),
    STATE_CONFIRMING: (_OFF,   _FAST,  _OFF),
    STATE_LOCKED:     (_OFF,   _OFF,   _SOLID),
    STATE_DETOUR:     (_OFF,   _SLOW,  _SOLID),
    STATE_IMPASSABLE: (_SOLID, _OFF,   _OFF),
}
_DEFAULT_LIGHTS = (_SLOW, _OFF, _OFF)      # 未知狀態比照 waiting


def _phase(mode, now):
    """把閃爍模式換算成這一瞬間的亮滅。

    用時間相位直接算,不開執行緒 —— 開執行緒會拖慢 camera callback。
    """
    if mode == _SOLID:
        return True
    if mode == _SLOW:
        return (now % SLOW_PERIOD) < (SLOW_PERIOD / 2.0)
    if mode == _FAST:
        return (now % FAST_PERIOD) < (FAST_PERIOD / 2.0)
    return False


class PrintBackend(object):
    """沒有硬體時的替身,讓筆電上也能跑測試。"""

    name = 'print'
    distance_m = None

    def set_lights(self, red, yellow, green):
        pass

    def beep(self, seconds):
        print('[buzzer] beep (lock acquired)')

    def close(self):
        pass


class GpioBackend(object):
    """三色燈 / 蜂鳴器直接接 Jetson Nano 的 40-pin GPIO。

    備案路線。**沒有測距功能** —— VL53L0X 已定案接在 Arduino 上,
    走這條的話 distance_m 永遠是 None,融合邏輯會單獨採信相機。
    """

    name = 'gpio'
    distance_m = None

    def __init__(self, red_pin=RED_PIN, yellow_pin=YELLOW_PIN,
                 green_pin=GREEN_PIN, buzzer_pin=BUZZER_PIN):
        import Jetson.GPIO as GPIO
        self.GPIO = GPIO
        self.pins = (red_pin, yellow_pin, green_pin)
        self.buzzer_pin = buzzer_pin
        self._beep_timer = None
        GPIO.setmode(GPIO.BOARD)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(buzzer_pin, GPIO.OUT, initial=GPIO.LOW)

    def set_lights(self, red, yellow, green):
        for pin, on in zip(self.pins, (red, yellow, green)):
            level = on if ACTIVE_HIGH else (not on)
            self.GPIO.output(pin, self.GPIO.HIGH if level else self.GPIO.LOW)

    def beep(self, seconds):
        if self._beep_timer is not None:
            self._beep_timer.cancel()
        self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH)
        self._beep_timer = threading.Timer(seconds, self._buzzer_off)
        self._beep_timer.daemon = True
        self._beep_timer.start()

    def _buzzer_off(self):
        self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)

    def close(self):
        if self._beep_timer is not None:
            self._beep_timer.cancel()
        self.set_lights(False, False, False)
        self._buzzer_off()
        self.GPIO.cleanup(list(self.pins) + [self.buzzer_pin])


class ArduinoBackend(object):
    """紅綠燈 / 蜂鳴器 / VL53L0X 全部接在 Arduino 上,透過 USB 序列埠溝通。

    協定見 arduino_feedback/arduino_feedback.ino 檔頭。

    **這支類別是序列埠的唯一擁有者。** 它開一條背景執行緒,把 Arduino 送過來的
    每一行都讀掉並分類:`D:<mm>` 存成距離,其餘(指令回覆)丟棄。
    rangefinder.py 只向它要數字,不另外開連線。

    為什麼一定要有這條讀取執行緒:
      - Arduino 以 10Hz 主動推播距離。沒人讀的話,Linux 端的接收緩衝區會塞滿,
        之後的讀取拿到的都是過期資料
      - 改成「問一句答一句」的話,camera callback 每幀都要等 Arduino 回覆,
        在 115200 下約 2ms,10 FPS 就是每秒白白等掉 20ms
    """

    name = 'arduino'

    def __init__(self, port=ARDUINO_PORT, baud=ARDUINO_BAUD):
        import serial
        # write_timeout 一定要設:Arduino 沒讀走資料時,寫入會卡住 camera callback
        self.ser = serial.Serial(port, baud, timeout=0.2, write_timeout=0.2)
        self.port = port
        self.sensor_ok = None          # None = 還不知道;由 READY / PONG 訊息填入

        self._samples = deque(maxlen=DISTANCE_WINDOW)   # [(時間, 公尺)]
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._running = True

        # 開啟序列埠會透過 DTR 讓 Arduino 重開機,不等它就會漏掉前幾個指令
        time.sleep(ARDUINO_RESET_WAIT)
        self.ser.reset_input_buffer()

        self._reader = threading.Thread(target=self._read_loop)
        self._reader.daemon = True
        self._reader.start()

    # -- 背景讀取 ---------------------------------------------------------
    def _read_loop(self):
        while self._running:
            try:
                raw = self.ser.readline()
            except Exception:
                # 拔線、埠被搶走等等。不該讓整台車停擺,睡一下再試。
                time.sleep(0.5)
                continue
            if not raw:
                continue                     # timeout,正常現象
            try:
                line = raw.decode('ascii', 'ignore').strip()
            except Exception:
                continue
            self._handle_line(line)

    def _handle_line(self, line):
        if line.startswith('D:'):
            try:
                mm = int(line[2:])
            except ValueError:
                return
            if mm < 0:
                return                       # -1 = 超出量程或讀取失敗,不是 0 公尺
            with self._lock:
                self._samples.append((time.time(), mm / 1000.0))
        elif line.startswith('READY') or line.startswith('PONG'):
            self.sensor_ok = ('vl53l0x=ok' in line)
        elif line.startswith('ERR'):
            print('[arduino] %s' % line)

    # -- 距離 -------------------------------------------------------------
    @property
    def distance_m(self):
        """回傳中位數濾波後的距離(公尺),沒有新鮮讀數時回 None。

        用中位數不用平均:雷射偶爾會吐出一個離群值(反射不良、邊緣繞射),
        平均會被它拉走,中位數不會。
        """
        cutoff = time.time() - DISTANCE_STALE_SECONDS
        with self._lock:
            fresh = [d for t, d in self._samples if t >= cutoff]
        if not fresh:
            return None
        fresh.sort()
        return fresh[len(fresh) // 2]

    # -- 輸出 -------------------------------------------------------------
    def _send(self, command):
        try:
            with self._write_lock:           # 避免兩個執行緒的指令交錯
                self.ser.write((command + '\n').encode('ascii'))
        except Exception as e:
            # 序列埠出問題不該讓整台車停擺,印一次就繼續
            print('[arduino] 指令 %r 送出失敗: %s' % (command, e))

    def set_lights(self, red, yellow, green):
        if not ACTIVE_HIGH:
            red, yellow, green = (not red), (not yellow), (not green)
        self._send('LIGHT:%d%d%d' % (bool(red), bool(yellow), bool(green)))

    def beep(self, seconds):
        # 蜂鳴器的計時放在 Arduino 端,所以這裡送完就走,不開執行緒也不會阻塞
        self._send('BEEP:%d' % int(seconds * 1000))

    def ping(self):
        """送出連線檢查。回覆由背景執行緒收下並更新 sensor_ok。"""
        self._send('PING')

    def close(self):
        self.set_lights(False, False, False)
        self._running = False
        try:
            self._reader.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


def _make_backend():
    """依 BACKEND 建立後端,失敗就退回印字模式(例如在筆電上)。"""
    if BACKEND == 'arduino':
        try:
            backend = ArduinoBackend()
            print('[feedback] 已連上 Arduino:%s @ %d' % (backend.port, ARDUINO_BAUD))
            backend.ping()
            time.sleep(0.2)                  # 等背景執行緒收下 PONG
            if backend.sensor_ok is True:
                print('[feedback] VL53L0X:已就緒')
            elif backend.sensor_ok is False:
                print('[feedback] VL53L0X:未偵測到(紅綠燈與蜂鳴器不受影響)')
            else:
                print('[feedback] VL53L0X:沒收到回覆,確認韌體鮑率是否為 %d' % ARDUINO_BAUD)
            return backend
        except Exception as e:
            print('[feedback] 連不上 Arduino(%s),改用印字模式' % e)
            return PrintBackend()
    if BACKEND == 'gpio':
        try:
            return GpioBackend()
        except Exception as e:
            print('[feedback] Jetson.GPIO 不可用(%s),改用印字模式' % e)
            return PrintBackend()
    return PrintBackend()


class Feedback(object):

    def __init__(self, backend=None):
        self.backend = backend or _make_backend()
        self._lights = None      # None = 尚未寫過,強制第一次一定送值

    # --- 紅綠燈:依狀態決定亮滅 -------------------------------------------
    def update_lights(self, state, now):
        """每幀呼叫。對照表見本檔上方的 STATE_LIGHTS。"""
        modes = STATE_LIGHTS.get(state, _DEFAULT_LIGHTS)
        combo = tuple(_phase(mode, now) for mode in modes)
        self._write(combo)

    # 舊名保留:notebook 或測試若還寫著 update_led 也不會壞
    update_led = update_lights

    def _write(self, combo):
        # 只在組合改變時才真的送指令。走序列埠時這點特別重要:
        # 每幀都送會塞爆 Arduino 的接收緩衝區。
        if combo == self._lights:
            return
        self._lights = combo
        self.backend.set_lights(*combo)

    # --- 蜂鳴器:只有 lock_acquired 會呼叫這裡 -----------------------------
    def beep_locked(self):
        self.backend.beep(BEEP_SECONDS)

    # --- 距離:轉給後端,沒有測距能力的後端回 None -------------------------
    @property
    def distance_m(self):
        return getattr(self.backend, 'distance_m', None)

    def close(self):
        self._lights = None
        self.backend.close()


# --- 本機事件記錄 ---------------------------------------------------------
#
# ⚠️ 2026-09-15 架構定案:車上**不再直接推播 LINE**。
#
# 車上只把狀態透過 MQTT 送到車外電腦(見 notebook 的 report_to_caregiver),
# 由車外的異常判定狀態機決定要不要通知家屬、以及訊息怎麼寫。
# 若兩邊都推,照護者會收到重複訊息。
#
# 所以底下這組函式現在的定位是**本機除錯用的事件記錄**,預設只印在 console。
# set_sender() 仍然保留,方便需要時把事件導到別的地方(例如寫檔)。

def _default_sender(message):
    print('[notify] ' + message)


_sender = _default_sender
_last_sent = {}


def set_sender(fn):
    """fn(message: str) -> None。會在背景執行緒被呼叫,實作端請自行處理逾時。"""
    global _sender
    _sender = fn


def notify_caregiver(message, key=None, cooldown=0.0):
    """非同步送出,避免 HTTP 逾時把 camera callback 卡住。

    key / cooldown:同一個 key 在 cooldown 秒內只送一次。
    「鎖定確認中」這種訊息會隨目標走動反覆觸發,一定要節流,否則會洗爆照護者。
    """
    if key is not None and cooldown > 0:
        now = time.time()
        last = _last_sent.get(key)
        if last is not None and (now - last) < cooldown:
            return
        _last_sent[key] = now
    t = threading.Thread(target=_send_safely, args=(message,))
    t.daemon = True
    t.start()


def _send_safely(message):
    try:
        _sender(message)
    except Exception as e:
        print('[notify] 傳送失敗: %s' % e)
