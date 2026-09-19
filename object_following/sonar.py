"""HC-SR04 超音波測距(前方單顆)。

===========================================================================
 ⚠️ 已停用(2026-09-18)。本檔案僅作為報告的對照資料保留,不要 import。
===========================================================================

 現在用的是 rangefinder.py + VL53L0X 雷射 ToF,接在 Arduino 的 I2C 上。
 整個專案已經沒有任何地方 import 這支檔案。

 換掉的三個理由:
   1. 下面那套「ECHO 接 Jetson GPIO 要串分壓電阻」的接線不再適用。
      VL53L0X 板上有穩壓,不用電阻,原本燒 Jetson 的風險整個消失
   2. 下面 _measure_once() 那種「送脈衝 → 等回波 → 掐秒錶」的做法,
      在非即時作業系統上會被排程打斷。雷射的測距與計時都在晶片內部完成
   3. 超音波遇到衣服、毛衣、海綿會吸音 —— 對「跟著一個長輩」特別不利

 ⚠️ 底下的接線說明是**歷史紀錄,不是現在的接線指引**。
    照著接不會壞,但接了也沒有程式會去讀它。
    現行接線見 README.md 的「接線(全部接在 Arduino 上)」章節。

---------------------------------------------------------------------------
 以下為 2026-09-13 的原始內容,未修改
---------------------------------------------------------------------------

⚠️ 接線務必先看 README 的「HC-SR04 接線」章節:
   ECHO 腳輸出 5V,Jetson Nano 的 GPIO 只耐 3.3V,**必須經分壓電阻**,
   直接接會燒掉 Jetson。

為什麼要開背景執行緒:
  一次測距最久要等 timeout(2 公尺約 12ms),而 camera callback 在 10~15 FPS 下
  每幀只有 66~100ms 的預算,裡面還要跑兩個神經網路。若在 callback 裡同步量測,
  等於白白吃掉超過一成的時間預算,還會被 Linux 排程抖動放大。
  這裡改成背景固定頻率量測,callback 只讀最新值,不阻塞。

在筆電上 import 不會炸(抓不到 Jetson.GPIO 時 distance 永遠回 None)。
"""

import threading
import time
from collections import deque

# TODO: 依實際接線確認。BOARD 編號(40-pin 排針的實體腳位序號)。
TRIG_PIN = 16
ECHO_PIN = 22

SOUND_SPEED = 343.0        # m/s,約 20°C
MAX_DISTANCE_M = 2.0       # 超過此距離視為無回波
SAMPLE_HZ = 10.0           # 背景量測頻率
MEDIAN_WINDOW = 5          # 中位數濾波的樣本數
STALE_SECONDS = 1.0        # 超過這麼久沒有有效讀數就回 None

try:
    import Jetson.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    GPIO = None
    _HAS_GPIO = False


class Sonar(object):

    def __init__(self, trig_pin=TRIG_PIN, echo_pin=ECHO_PIN,
                 sample_hz=SAMPLE_HZ, max_distance_m=MAX_DISTANCE_M):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.max_distance_m = max_distance_m
        self._interval = 1.0 / sample_hz
        # 回波往返 timeout,再留 20% 餘裕
        self._timeout = (2.0 * max_distance_m / SOUND_SPEED) * 1.2

        self._samples = deque(maxlen=MEDIAN_WINDOW)
        self._last_valid_at = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        if _HAS_GPIO:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.trig_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.echo_pin, GPIO.IN)
            self.start()

    # --- 對外介面 ---------------------------------------------------------
    @property
    def distance(self):
        """最新的距離(公尺)。沒有有效讀數(含沒有 GPIO 的環境)時回 None。

        回 None 代表「不知道」,呼叫端必須把它當成沒有資訊,
        不可以當成「前方淨空」——那會在感測器故障時直接撞上去。
        """
        with self._lock:
            if not self._samples or self._last_valid_at is None:
                return None
            if (time.time() - self._last_valid_at) > STALE_SECONDS:
                return None
            ordered = sorted(self._samples)
            return ordered[len(ordered) // 2]     # 中位數,擋掉偶發亂跳

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def close(self):
        self.stop()
        if _HAS_GPIO:
            GPIO.cleanup([self.trig_pin, self.echo_pin])

    # --- 背景量測 ---------------------------------------------------------
    def _loop(self):
        while self._running:
            distance = self._measure_once()
            if distance is not None:
                with self._lock:
                    self._samples.append(distance)
                    self._last_valid_at = time.time()
            time.sleep(self._interval)

    def _measure_once(self):
        """送一個 10us 觸發脈衝,量 ECHO 的高電位長度。無回波回 None。"""
        if not _HAS_GPIO:
            return None

        GPIO.output(self.trig_pin, GPIO.HIGH)
        time.sleep(0.00001)                     # 10 微秒
        GPIO.output(self.trig_pin, GPIO.LOW)

        deadline = time.time() + self._timeout
        while GPIO.input(self.echo_pin) == 0:   # 等回波開始
            if time.time() > deadline:
                return None
        rise = time.time()

        deadline = rise + self._timeout
        while GPIO.input(self.echo_pin) == 1:   # 等回波結束
            if time.time() > deadline:
                return None
        fall = time.time()

        distance = (fall - rise) * SOUND_SPEED / 2.0
        if distance <= 0 or distance > self.max_distance_m:
            return None
        return distance


def is_blocked(prob_blocked, distance_m, prob_threshold=0.5, distance_threshold=0.35):
    """把相機碰撞模型與超音波融合成單一的「被擋住」訊號。

    兩者給的是不同性質的資訊,不是同一件事的兩個估計值:

    - 相機的 AlexNet 碰撞模型:二分類,只回答「擋住/沒擋住」,沒有距離概念,
      而且判斷結果取決於訓練資料涵蓋了哪些場景
    - HC-SR04:在感測器架設高度上量出實際距離(公尺),但錐狀波束僅約 15 度,
      對斜面、布料、海綿等吸音或反射不良的表面會漏測

    因此用 OR 而不是加權平均:平均之後任一方的警告都會被另一方稀釋掉,結果是照樣撞上。
    這與需求文件「避障與跟隨為優先權覆蓋架構,不是加權混合」的安全考量一致。

    distance_m 為 None(未啟用、感測器故障或超出量程)時視為「沒有資訊」,
    單獨採信相機的判斷,不因此放行。
    """
    if prob_blocked > prob_threshold:
        return True
    if distance_m is not None and distance_m < distance_threshold:
        return True
    return False
