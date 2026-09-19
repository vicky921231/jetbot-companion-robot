"""前方測距與「被擋住」訊號的融合。

2026-09-18 取代 sonar.py。兩個變動:

  1. **HC-SR04 超音波 → VL53L0X 雷射 ToF**(GY-530)
  2. **接在 Arduino 上,不接 Jetson GPIO**

換掉之後消失的三個問題:
  - HC-SR04 的 ECHO 輸出 5V,Jetson GPIO 只耐 3.3V,**必須串分壓電阻,接錯燒板**
    → VL53L0X 走 I2C 且板上有穩壓,不用電阻
  - 舊版要在 Python 裡「送脈衝 → 等回波 → 掐秒錶」,而 Linux 不是即時作業系統,
    那個迴圈隨時會被排程打斷,量出來的時間會抖
    → 現在測距與計時都在感測器晶片內部完成,Arduino 只是讀一個數字
  - 布料、衣服、海綿會吸音,對「跟著一個穿毛衣的長輩」特別不利
    → 雷射不管表面軟硬,只要會反光就量得到

**這支檔案不再開任何執行緒,也不碰硬體。** 量測在 Arduino 上以 10Hz 進行並主動
推播,feedback.ArduinoBackend 的背景執行緒收下並做中位數濾波。這裡只是個轉接頭,
外加下面那個融合函式 —— 真正有內容的是 is_blocked()。

用法(notebook):
    fb = feedback.Feedback()
    finder = rangefinder.RangeFinder(fb)
    ...
    blocked = rangefinder.is_blocked(prob_blocked, finder.distance)
"""


class RangeFinder(object):
    """把 feedback 後端的距離讀數包成與舊 Sonar 相同的介面。

    source 可以是 feedback.Feedback 或它底下的 backend,任何有 distance_m
    屬性的物件都行。沒有測距能力的後端(PrintBackend / GpioBackend)
    distance_m 永遠是 None,融合邏輯會自動單獨採信相機。
    """

    def __init__(self, source):
        self._source = source

    @property
    def distance(self):
        """公尺。沒有新鮮讀數(感測器沒接、超出量程、表面吸光)時回 None。"""
        return getattr(self._source, 'distance_m', None)

    def close(self):
        """什麼都不做 —— 序列埠由 feedback.Feedback 擁有,由它負責關閉。

        留著這個方法是為了讓 notebook 的收尾那一格不用改寫。
        """
        pass


def is_blocked(prob_blocked, distance_m, prob_threshold=0.5, distance_threshold=0.35):
    """把相機碰撞模型與雷射測距融合成單一的「被擋住」訊號。

    兩者給的是不同性質的資訊,不是同一件事的兩個估計值:

    - 相機碰撞模型:二分類,只回答「擋住/沒擋住」,沒有距離概念,
      判斷結果取決於訓練資料涵蓋了哪些場景
    - VL53L0X:量出實際距離(公尺),但**深色與黑色表面會吸收紅外線**,
      讀數會變短或直接失敗;視野角約 25 度,比 HC-SR04 的 15 度還寬,
      所以正前方以外的東西更容易混進讀數裡

    因此用 OR 而不是加權平均:兩者盲區不同,平均之後任一方的警告都會被另一方
    稀釋掉,結果是照樣撞上。這與需求文件「避障與跟隨為優先權覆蓋架構,
    不是加權混合」的安全考量一致。

    distance_m 為 None(未接線、感測器故障、超出量程或表面吸光)時視為
    「沒有資訊」,單獨採信相機的判斷,**不因此放行**。

    ⚠️ 呼叫端要負責的事:若正前方就是鎖定中的目標,近距離代表「人很近」而不是
       「有障礙物」,這時要傳 distance_m=None。判斷式是 sm.expects_target_ahead(),
       notebook 的 execute() 已經處理。少了那一行,機器人會把被跟隨的人繞開。
    """
    if prob_blocked > prob_threshold:
        return True
    if distance_m is not None and distance_m < distance_threshold:
        return True
    return False
