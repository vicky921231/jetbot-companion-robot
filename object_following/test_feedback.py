"""紅綠燈邏輯與 Arduino 協定的離線測試。在筆電上直接跑,不需要硬體:

    python test_feedback.py

測的是「狀態 → 三顆燈」的對照、去重、指令格式、以及序列埠回傳行的解析。
實體接線對不對是 test_hardware.py 的事(那支要在 JetBot 上跑,而且要有人看)。
"""

import time

import feedback
from follow_state import (
    STATE_WAITING, STATE_CONFIRMING, STATE_LOCKED, STATE_DETOUR, STATE_IMPASSABLE,
)


class FakeBackend(object):
    """記下每一次輸出,不碰任何硬體。"""

    name = 'fake'
    distance_m = None

    def __init__(self):
        self.lights = []
        self.beeps = []
        self.closed = False

    def set_lights(self, red, yellow, green):
        self.lights.append((red, yellow, green))

    def beep(self, seconds):
        self.beeps.append(seconds)

    def close(self):
        self.closed = True


def check(name, cond):
    print(('  PASS  ' if cond else '  FAIL  ') + name)
    if not cond:
        raise AssertionError(name)


# 選兩個相位:0.0 = 所有閃爍都在「亮」的半週期;0.7 = 慢閃滅、快閃也滅
ON_PHASE = 0.0
OFF_PHASE = 0.7


def lights_at(state, now):
    be = FakeBackend()
    fb = feedback.Feedback(be)
    fb.update_lights(state, now)
    return be.lights[-1]


def test_state_table():
    print('狀態對照表(相位落在「亮」的半週期)')
    check('waiting    → 紅', lights_at(STATE_WAITING, ON_PHASE) == (True, False, False))
    check('confirming → 黃', lights_at(STATE_CONFIRMING, ON_PHASE) == (False, True, False))
    check('locked     → 綠', lights_at(STATE_LOCKED, ON_PHASE) == (False, False, True))
    check('detour     → 黃+綠', lights_at(STATE_DETOUR, ON_PHASE) == (False, True, True))
    check('impassable → 紅', lights_at(STATE_IMPASSABLE, ON_PHASE) == (True, False, False))


def test_blink_phases():
    print('閃爍相位')
    check('waiting 在滅的半週期會滅',
          lights_at(STATE_WAITING, OFF_PHASE) == (False, False, False))
    check('locked 恆亮,任何相位都亮',
          lights_at(STATE_LOCKED, OFF_PHASE) == (False, False, True))
    check('detour 的綠燈恆亮,只有黃燈在閃',
          lights_at(STATE_DETOUR, OFF_PHASE) == (False, False, True))
    check('confirming 快閃週期比 waiting 短',
          feedback.FAST_PERIOD < feedback.SLOW_PERIOD)


def test_detour_matches_locked_green():
    print('設計約束:繞行期間「還跟著你」的綠燈訊息不能改變')
    for now in (0.0, 0.3, 0.7, 1.4, 2.9):
        locked_green = lights_at(STATE_LOCKED, now)[2]
        detour_green = lights_at(STATE_DETOUR, now)[2]
        check('  相位 %.1f:兩者綠燈相同' % now, locked_green == detour_green)


def test_waiting_and_impassable_are_distinguishable():
    print('三色才解決的缺陷:waiting 與 impassable 不能長得一樣')
    same = all(lights_at(STATE_WAITING, t) == lights_at(STATE_IMPASSABLE, t)
               for t in (0.0, 0.3, 0.7, 1.4))
    check('兩個狀態的燈號不完全相同', not same)


def test_deduplication():
    print('去重:組合沒變就不重送(序列埠會被塞爆)')
    be = FakeBackend()
    fb = feedback.Feedback(be)
    for _ in range(50):
        fb.update_lights(STATE_LOCKED, 1.0)      # 同一個相位,組合不變
    check('50 次呼叫只送出 1 次', len(be.lights) == 1)

    fb.update_lights(STATE_WAITING, ON_PHASE)
    check('狀態改變就送', len(be.lights) == 2)


def test_beep_and_close():
    print('蜂鳴器與收尾')
    be = FakeBackend()
    fb = feedback.Feedback(be)
    fb.beep_locked()
    check('beep 長度 = BEEP_SECONDS', be.beeps == [feedback.BEEP_SECONDS])
    fb.close()
    check('close 有傳下去', be.closed)
    check('distance_m 沒有測距能力時回 None', fb.distance_m is None)


def _bare_arduino():
    """不開序列埠就建出 ArduinoBackend,只為了測解析邏輯。"""
    from collections import deque
    import threading
    be = object.__new__(feedback.ArduinoBackend)
    be._samples = deque(maxlen=feedback.DISTANCE_WINDOW)
    be._lock = threading.Lock()
    be.sensor_ok = None
    return be


def test_serial_line_parsing():
    print('Arduino 回傳行的解析')
    be = _bare_arduino()

    be._handle_line('READY vl53l0x=ok')
    check('READY 認出感測器就緒', be.sensor_ok is True)
    be._handle_line('PONG vl53l0x=absent')
    check('PONG 認出感測器不在', be.sensor_ok is False)

    be._handle_line('D:900')
    check('D:900 → 0.9 公尺', abs(be.distance_m - 0.9) < 1e-6)

    be._handle_line('D:-1')
    check('D:-1 是「讀不到」不是 0 公尺,要被忽略',
          abs(be.distance_m - 0.9) < 1e-6)

    be._handle_line('D:abc')
    check('壞掉的行不會炸也不會改變讀數', abs(be.distance_m - 0.9) < 1e-6)

    be._handle_line('LIGHT 100')          # 指令回覆,丟棄即可
    check('指令回覆不影響距離', abs(be.distance_m - 0.9) < 1e-6)


def test_median_filter():
    print('中位數濾波:雷射偶爾吐出的離群值不該拉走讀數')
    be = _bare_arduino()
    for mm in (800, 810, 1950, 805, 795):   # 1950 是離群值
        be._handle_line('D:%d' % mm)
    d = be.distance_m
    check('中位數落在真實值附近(實得 %.3f m)' % d, 0.79 < d < 0.82)


def test_staleness():
    print('過期偵測:感測器停止推播時要回 None,不能拿舊值當真')
    be = _bare_arduino()
    be._handle_line('D:900')
    check('剛收到時讀得到', be.distance_m is not None)
    with be._lock:
        stale = time.time() - feedback.DISTANCE_STALE_SECONDS - 1.0
        be._samples[0] = (stale, 0.9)
    check('超過 %.1f 秒沒更新就回 None' % feedback.DISTANCE_STALE_SECONDS,
          be.distance_m is None)


def test_light_command_format():
    print('指令格式:必須與 arduino_feedback.ino 的解析對得上')
    sent = []
    be = _bare_arduino()
    be._send = sent.append

    feedback.ArduinoBackend.set_lights(be, True, False, False)
    check('紅燈 → LIGHT:100', sent[-1] == 'LIGHT:100')
    feedback.ArduinoBackend.set_lights(be, False, True, True)
    check('黃+綠 → LIGHT:011', sent[-1] == 'LIGHT:011')
    feedback.ArduinoBackend.set_lights(be, False, False, False)
    check('全滅 → LIGHT:000', sent[-1] == 'LIGHT:000')

    feedback.ArduinoBackend.beep(be, 0.15)
    check('beep 0.15 秒 → BEEP:150', sent[-1] == 'BEEP:150')


if __name__ == '__main__':
    tests = [
        test_state_table,
        test_blink_phases,
        test_detour_matches_locked_green,
        test_waiting_and_impassable_are_distinguishable,
        test_deduplication,
        test_beep_and_close,
        test_serial_line_parsing,
        test_median_filter,
        test_staleness,
        test_light_command_format,
    ]
    for t in tests:
        t()
    print('\n全部通過(%d 組)' % len(tests))
