"""紅綠燈 / 蜂鳴器 / 雷射測距硬體測試 —— 在 JetBot 上執行。

這支刻意**不碰相機、不碰馬達、不載入任何神經網路**,只驗證輸出與感測器接對了沒。
接完線第一個跑這支,確認全部正常,再去跑完整的 notebook。

2026-09-18 改版:單顆 LED → 三色號誌;新增 VL53L0X 測距測試;鮑率 115200。

用法(JetBot 的 bash 終端機):
    cd ~/Notebooks/object_following
    python3 test_hardware.py

或在 JetBot 的 Jupyter cell 裡:
    %run test_hardware.py
"""

import time

import feedback
from follow_state import (
    STATE_WAITING, STATE_CONFIRMING, STATE_LOCKED, STATE_DETOUR, STATE_IMPASSABLE,
)


def ask(question):
    """回答 y 才算通過。"""
    return input(question + ' [y/n] ').strip().lower().startswith('y')


def hold(fb, state, seconds, note):
    """維持某個狀態一段時間,讓人眼看得出閃爍頻率。"""
    print('\n>>> ' + note)
    end = time.time() + seconds
    while time.time() < end:
        fb.update_lights(state, time.time())
        time.sleep(0.02)          # 模擬 50 FPS 的 callback 呼叫頻率


def show_only(fb, red, yellow, green, seconds):
    """繞過狀態對照表,直接點亮指定的燈 —— 用來確認三條線各自接對。"""
    fb._write((red, yellow, green))
    time.sleep(seconds)


def main():
    print('=' * 60)
    print('紅綠燈 / 蜂鳴器 / 雷射測距 硬體測試')
    print('設定的後端:%s   鮑率:%d' % (feedback.BACKEND, feedback.ARDUINO_BAUD))
    print('=' * 60)

    fb = feedback.Feedback()
    print('實際使用的後端:', fb.backend.name)

    if fb.backend.name == 'print':
        print('\n[!] 現在是印字模式,沒有真的驅動硬體。')
        if feedback.BACKEND == 'arduino':
            print('    連不上 Arduino。在 JetBot 的 bash 檢查:')
            print('      ls /dev/ttyACM* /dev/ttyUSB*')
            print('    找到實際的埠之後,改 feedback.py 的 ARDUINO_PORT。')
            print('    若出現 permission denied,執行:')
            print('      sudo usermod -aG dialout $USER   (需重新登入)')
        return

    results = []

    try:
        # --- 步驟 0:連線 ---------------------------------------------------
        if fb.backend.name == 'arduino':
            print('\n[步驟 0] Arduino 連線測試')
            fb.backend.ping()
            time.sleep(0.5)              # 等背景執行緒收下 PONG
            alive = fb.backend.sensor_ok is not None
            results.append(('Arduino 連線', alive))
            if alive:
                print('    收到回覆。VL53L0X:%s'
                      % ('已就緒' if fb.backend.sensor_ok else '未偵測到'))
            else:
                print('    → 沒收到回覆。最可能的兩個原因:')
                print('      (a) 還在跑舊版韌體 —— 舊版是 9600,現在是 %d'
                      % feedback.ARDUINO_BAUD)
                print('      (b) 韌體還沒燒錄。燒 arduino_feedback/arduino_feedback.ino')

        # --- 步驟 1:三條線各自接對了嗎 --------------------------------------
        # 先一顆一顆點,接錯線在這裡就會現形,不用等到看閃爍模式才發現。
        print('\n[步驟 1] 三顆燈逐一點亮')
        for name, combo in (('紅', (1, 0, 0)), ('黃', (0, 1, 0)), ('綠', (0, 0, 1))):
            print('\n>>> 現在應該只有【%s燈】亮' % name)
            show_only(fb, *combo, seconds=2.0)
            ok = ask('只有%s燈亮著,其他兩顆都是滅的嗎?' % name)
            results.append(('%s燈接線正確' % name, ok))
            if not ok:
                print('    → 對照 arduino_feedback.ino 的腳位:紅 D10 / 黃 D9 / 綠 D8')
                print('      三條訊號線接錯位置的話,把線調換即可,不用改程式。')
        show_only(fb, 0, 0, 0, 0.5)

        # --- 步驟 2:極性 ---------------------------------------------------
        # 這步決定 feedback.py 的 ACTIVE_HIGH 要不要改成 False。
        print('\n[步驟 2] 極性確認')
        if not any(ok for name, ok in results if '接線正確' in name):
            print('    三顆都沒亮 → 可能是共陽模組(低電位亮)。')
            print('    把 feedback.py 的 ACTIVE_HIGH 改成 False 再跑一次。')

        # --- 步驟 3:五個狀態的燈號 ------------------------------------------
        print('\n[步驟 3] 狀態對照表')
        hold(fb, STATE_WAITING, 5.0, 'waiting —— 應為【紅燈慢閃】(約 1 秒一次)')
        results.append(('waiting 紅慢閃', ask('看到紅燈慢速閃爍嗎?')))

        hold(fb, STATE_CONFIRMING, 5.0, 'confirming —— 應為【黃燈快閃】(約 0.2 秒一次)')
        results.append(('confirming 黃快閃', ask('看到黃燈、而且明顯比剛才快嗎?')))

        hold(fb, STATE_LOCKED, 4.0, 'locked —— 應為【綠燈恆亮】')
        results.append(('locked 綠恆亮', ask('綠燈穩定恆亮、完全沒閃嗎?')))

        hold(fb, STATE_DETOUR, 5.0, 'detour —— 應為【綠燈恆亮 + 黃燈慢閃】')
        results.append(('detour 綠亮+黃閃', ask('綠燈一直亮著,同時黃燈在慢閃嗎?')))

        hold(fb, STATE_IMPASSABLE, 4.0, 'impassable_wait —— 應為【紅燈恆亮】')
        results.append(('impassable 紅恆亮', ask('紅燈恆亮、沒有閃嗎?')))

        # --- 步驟 4:完整鎖定流程 -------------------------------------------
        # 這是目標者實際會看到的畫面:紅閃 → 黃閃 → 綠亮 + 一聲。
        print('\n[步驟 4] 模擬完整鎖定流程')
        print('    依序演一次:等待 → 確認中 → 鎖定完成')
        time.sleep(1.0)
        hold(fb, STATE_WAITING, 3.0, '目標還沒站定…(紅燈慢閃)')
        hold(fb, STATE_CONFIRMING, 0.75, '目標站定了,倒數 0.75 秒…(黃燈快閃)')
        fb.beep_locked()
        hold(fb, STATE_LOCKED, 3.0, '鎖定完成!(綠燈亮起,同時蜂鳴器短鳴一聲)')

        beeped = ask('聽到「一聲」短鳴嗎?')
        results.append(('蜂鳴器短鳴', beeped))
        if not beeped:
            print('    → 可能原因:')
            print('      (a) 蜂鳴器還沒接(D11)')
            print('      (b) 是「無源」蜂鳴器 —— 固定高電位不會響,需要 PWM 方波。')
            print('          把 .ino 的 ACTIVE_BUZZER 改成 false 再燒一次')
            print('      (c) 腳位不對')

        # --- 步驟 5:雷射測距 ------------------------------------------------
        print('\n[步驟 5] VL53L0X 雷射測距')
        if fb.backend.name != 'arduino':
            print('    跳過:測距接在 Arduino 上,目前後端是 %s' % fb.backend.name)
        elif fb.backend.sensor_ok is False:
            print('    韌體回報 vl53l0x=absent —— 感測器沒接或接錯。')
            print('    接線:VIN→5V、GND→GND、SCL→A5、SDA→A4')
            results.append(('VL53L0X 偵測到', False))
        else:
            print('    把手掌放在感測器前面慢慢前後移動,看數字有沒有跟著變。')
            print('    (讀數每 0.1 秒更新一次,共顯示 10 秒)')
            seen = []
            end = time.time() + 10.0
            while time.time() < end:
                d = fb.distance_m
                print('      距離:%s' % ('讀不到' if d is None else '%.3f m' % d))
                if d is not None:
                    seen.append(d)
                time.sleep(0.5)
            if not seen:
                print('    完全讀不到數字。')
                results.append(('VL53L0X 有讀數', False))
            else:
                spread = max(seen) - min(seen)
                print('    收到 %d 筆,範圍 %.3f ~ %.3f m' % (len(seen), min(seen), max(seen)))
                results.append(('VL53L0X 有讀數', True))
                results.append(('VL53L0X 讀數會隨距離變化', spread > 0.05))
                if spread <= 0.05:
                    print('    → 數字幾乎沒變。手有真的移動嗎?')
                    print('      若確實有動:深色/黑色表面會吸收紅外線,換淺色物體再試。')

    finally:
        fb.close()
        print('\n序列埠已釋放。')

    # --- 結果 --------------------------------------------------------------
    print('\n' + '=' * 60)
    for name, ok in results:
        print(('  PASS  ' if ok else '  FAIL  ') + name)
    failed = [n for n, ok in results if not ok]
    if failed:
        print('\n未通過:' + '、'.join(failed))
        print('把這份輸出貼回去,就能對症修改。')
    else:
        print('\n全部通過,可以進行下一步:跑完整的 notebook。')
        print('⚠️ 跑 notebook 前務必把車輪架空 —— 鎖定成功的瞬間馬達會立刻轉動。')
    print('=' * 60)


if __name__ == '__main__':
    main()
