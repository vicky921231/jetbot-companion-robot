"""目標身分標記偵測(v1:螢光雙色貼紙 + HSV 色域門檻)。

取代原本規劃的 re-ID 外觀比對。做法是在被跟隨者**兩腿的小腿外側各貼一張**
「上桃紅 / 下螢光黃」的雙色標記,用 HSV 門檻找出來。

為什麼不用神經網路(YOLO):
  這件事只要判斷「畫面上有沒有這兩個特定顏色、而且相鄰」,是顏色查表問題,
  不是辨識問題。HSV 門檻約 1~2ms、零訓練資料;YOLO 要 30~50ms 加上幾百張
  手工標框,而且 JetPack 4.5 的 Python 3.6 裝不了 ultralytics。
  兩個神經網路(SSD 人物偵測 + ResNet18 避障)已經吃滿 FPS 預算,不能再加第三個。

為什麼貼兩條腿:
  單腿的標記在走路時會被另一條腿週期性遮住(約 1Hz)。標記一消失,
  identity_score 回 0 → select_target() 回 None → _handle_lost() → 車子停住,
  結果是走兩步停一下。兩條腿各貼一張,任何步態都至少看得到一張。
  遮擋仍會發生,所以另外加了 grace_seconds 寬限。

這支檔案只 import cv2 / numpy,**不碰 jetbot / torch**,所以可以在筆電上直接
拿照片測試(見 marker_calibration/)。

接法(notebook 三行):
    import marker
    sm.identity_score = marker.MarkerVerifier()
    config.identity_threshold = 0.5
"""

import time
from collections import namedtuple

import cv2
import numpy as np


class MarkerConfig(object):
    """所有可調參數集中在這裡。跑過 marker_calibration/analyze_marker.py 之後,
    用它印出來的建議值取代下面的顏色門檻。
    """

    # --- 顏色門檻 ---------------------------------------------------------
    # OpenCV 的 HSV 是 H 0~179、S 0~255、V 0~255,
    # 不是一般繪圖軟體的 H 0~360、S 0~100、V 0~100。抄色票時要先換算。
    #
    # 下面是**估計的起點,不是實測值**。務必跑過校正腳本再定案。
    #
    # 寫成 list 是為了處理色相環繞:桃紅若偏紅而跨過 H=179/0 的接縫,
    # 就要拆成兩段,例如
    #   [((170, 150, 90), (179, 255, 255)), ((0, 150, 90), (6, 255, 255))]

    # 桃紅粉 —— 主鑑別色。這個色相在室內環境幾乎不存在(沒有這個顏色的牆、
    # 地板、家具、皮膚),所以它負責回答「這是標記」。
    pink_ranges = [((158, 150, 90), (176, 255, 255))]

    # 螢光黃 —— 確認色。它的色相(H 22~42)離膚色(H 5~20)和木地板(H 10~25)
    # 很近,**光靠色相分不開,一定要靠高飽和度**:螢光色 S 通常 >200,
    # 膚色和木頭很少超過 150。所以 S 下限不能往下調。
    second_ranges = [((22, 150, 90), (42, 255, 255))]
    # 若改用螢光綠(離膚色更遠、室內更安全,除非現場有盆栽),換成:
    # second_ranges = [((40, 150, 90), (75, 255, 255))]

    # --- 搜尋範圍 ---------------------------------------------------------
    # 只找人框的下半部。標記貼在小腿,限制範圍能擋掉「粉紅色上衣」這類誤判。
    # 近距離(0.45m)時畫面只涵蓋地板到腰,人框會被上緣裁掉,但下半部仍是腿。
    search_bottom_fraction = 0.55

    # --- 面積門檻 ---------------------------------------------------------
    # ⚠️ 這些數字綁定畫面解析度。detect() 吃的是 live_demo 的**原始相機畫面**,
    #    目前是 300x300(官方值,因為 SSD MobileNet 吃 300x300 輸入),
    #    不是避障模型那條鏈上的 224x168。改相機解析度就要重新校正。
    #
    # 推導:相機擷取 816x616(4:3)縮放成 300x300,所以垂直/水平的
    # 每像素尺度不同。由 min_bbox_height=0.30 對應 1.5m 反推:
    #   垂直 1px 約 1.9cm、水平 1px 約 2.5cm
    #   15x20cm 的標記在 1.5m 處約 6x10 px,桃紅那半約 6x5 = 30 px
    #   0.8m(跟隨距離)處約 11x20 px,桃紅那半約 110 px
    # 門檻設在 1.5m 期望值的一半左右。校正後用 --check 印出的實測像素數重設。
    min_pink_area = 15
    # 反向防呆:桃紅佔搜尋區域超過這個比例,那不是標記,是一件桃紅色的褲子。
    max_pink_ratio = 0.35

    # --- 雙色驗證 ---------------------------------------------------------
    # 在桃紅色塊的上下各延伸這個倍率(以色塊高度為單位)去找第二個顏色。
    neighbourhood_scale = 1.5
    # 第二色面積要達到桃紅面積的這個比例才算數。
    second_area_ratio = 0.25

    # 距離自適應:只有夠近才要求雙色。框高 0.30 約等於 1.5m、0.45 約等於 1.0m。
    #
    # 在 300x300 下,1.5m 處整張標記約 6x10 px,切成上下兩塊各剩 6x5 px,
    # 其實勉強解析得出來 —— 所以 0.45 是**保守值**,實測可能可以放寬到 0.30
    # (也就是全程都要求雙色)。跑 --check 看「雙色驗證成功的最遠距離」再定。
    #
    # 遠距離退化成「只認桃紅」是可接受的:身分驗證最需要準的時候是跟隨中
    # (0.45~0.80m),而那個距離每塊有 11x10 px,雙色綽綽有餘。
    dual_color_min_bbox_h = 0.45

    # --- 遮擋寬限 ---------------------------------------------------------
    # 轉身、兩腿交錯、褲管蓋住造成的短暫消失,不該立刻判定跟丟。
    grace_seconds = 1.5


MarkerResult = namedtuple(
    'MarkerResult',
    ['found', 'pink_area', 'second_area', 'dual_required', 'bbox_px', 'clipped_ratio'])

EMPTY_RESULT = MarkerResult(False, 0, 0, False, None, 0.0)


# --- 內部工具 --------------------------------------------------------------

def _to_pixel_box(bbox, w, h):
    """正規化 bbox [x1,y1,x2,y2](0~1)轉成像素座標,並保證至少 1x1。"""
    x1 = int(round(max(0.0, min(1.0, bbox[0])) * w))
    y1 = int(round(max(0.0, min(1.0, bbox[1])) * h))
    x2 = int(round(max(0.0, min(1.0, bbox[2])) * w))
    y2 = int(round(max(0.0, min(1.0, bbox[3])) * h))
    if x2 <= x1:
        x1, x2 = max(0, x1 - 1), min(w, x1 + 1)
    if y2 <= y1:
        y1, y2 = max(0, y1 - 1), min(h, y1 + 1)
    return x1, y1, x2, y2


def _build_mask(hsv, ranges):
    """多段色域取聯集,用來處理色相環繞。"""
    mask = None
    for lo, hi in ranges:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    return mask


def _largest_blob(mask, min_area):
    """回傳 (面積, x, y, w, h),找不到夠大的回 None。

    刻意**不做形態學開運算**:3x3 的 kernel 會把 4px 寬的標記整個侵蝕掉。
    在這個解析度下,雜訊要靠面積門檻濾,不能靠形態學。

    用 connectedComponentsWithStats 而不是 contourArea:後者算的是多邊形面積,
    對 4x3 這種極小色塊會嚴重低估。
    """
    num, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_i, best_area = -1, 0
    for i in range(1, num):          # label 0 是背景
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_i, best_area = i, area
    if best_i < 0 or best_area < min_area:
        return None
    return (best_area,
            int(stats[best_i, cv2.CC_STAT_LEFT]),
            int(stats[best_i, cv2.CC_STAT_TOP]),
            int(stats[best_i, cv2.CC_STAT_WIDTH]),
            int(stats[best_i, cv2.CC_STAT_HEIGHT]))


def _clipped_ratio(hsv):
    """過曝比例:又亮又不飽和 = 被壓成白色,色相已經不可信。

    亮面材質在燈下產生鏡面高光時就會出現這種像素。校正時用來判斷
    該買霧面還是亮面。
    """
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    clipped = np.count_nonzero((v >= 250) & (s < 60))
    total = hsv.shape[0] * hsv.shape[1]
    return float(clipped) / total if total else 0.0


# --- 主要進入點 ------------------------------------------------------------

def detect(image, bbox, config=None, diagnostics=False):
    """在 image 的 bbox 範圍內找標記。

    image:       BGR numpy 陣列(相機原生格式,不用先轉)
    bbox:        正規化的人框 [x1, y1, x2, y2]
    diagnostics: True 才計算過曝比例(校正時用,上車時關掉省時間)
    """
    if image is None or getattr(image, 'size', 0) == 0:
        return EMPTY_RESULT

    cfg = config or MarkerConfig()
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _to_pixel_box(bbox, w, h)

    # 只取人框下半部。ROI 的座標是相對的,最後要加回 (x1, top) 才是全畫面座標。
    top = y1 + int((y2 - y1) * (1.0 - cfg.search_bottom_fraction))
    roi = image[top:y2, x1:x2]
    if roi.size == 0:
        return EMPTY_RESULT

    # 只轉 ROI 不轉整張:cvtColor 是逐像素運算,整張 224x168 比一小塊腿部貴得多
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    clipped = _clipped_ratio(hsv) if diagnostics else 0.0

    pink_mask = _build_mask(hsv, cfg.pink_ranges)
    blob = _largest_blob(pink_mask, cfg.min_pink_area)
    if blob is None:
        return EMPTY_RESULT._replace(clipped_ratio=clipped)

    pink_area, bx, by, bw, bh = blob

    roi_pixels = roi.shape[0] * roi.shape[1]
    if roi_pixels and float(pink_area) / roi_pixels > cfg.max_pink_ratio:
        # 桃紅佔滿了整個腿部區域 → 這是衣物,不是標記
        return EMPTY_RESULT._replace(pink_area=pink_area, clipped_ratio=clipped)

    abs_box = (x1 + bx, top + by, bw, bh)
    bbox_h = abs(bbox[3] - bbox[1])
    dual_required = bbox_h >= cfg.dual_color_min_bbox_h

    if not dual_required:
        # 遠距離:解析不出雙色結構,只認桃紅
        return MarkerResult(True, pink_area, 0, False, abs_box, clipped)

    # 近距離:在桃紅色塊的鄰域找第二個顏色。
    # 刻意不檢查「黃色一定在桃紅下方」—— 在 4~6px 的尺度下,質心的上下關係
    # 已經不可靠了。「兩個螢光色緊鄰」本身就是極罕見的巧合,這樣就夠了。
    pad_y = max(1, int(round(bh * cfg.neighbourhood_scale)))
    pad_x = max(1, int(round(bw * 0.5)))
    ny1 = max(0, by - pad_y)
    ny2 = min(hsv.shape[0], by + bh + pad_y)
    nx1 = max(0, bx - pad_x)
    nx2 = min(hsv.shape[1], bx + bw + pad_x)

    second_mask = _build_mask(hsv[ny1:ny2, nx1:nx2], cfg.second_ranges)
    second_area = int(cv2.countNonZero(second_mask))

    found = second_area >= cfg.second_area_ratio * pink_area
    return MarkerResult(found, pink_area, second_area, True, abs_box, clipped)


class MarkerVerifier(object):
    """接到 FollowStateMachine.identity_score 的介面上。

    簽章與 follow_state.default_identity_score 完全一致:
        (image, bbox, locked_target) -> 0.0 ~ 1.0
    搭配 FollowConfig.identity_threshold = 0.5 使用。
    """

    def __init__(self, config=None):
        self.config = config or MarkerConfig()
        self.last_result = EMPTY_RESULT
        self._last_ok = None

    def __call__(self, image, bbox, locked_target):
        result = detect(image, bbox, self.config)
        self.last_result = result

        if result.found:
            self._last_ok = time.time()
            return 1.0

        # 遮擋寬限**只在已經鎖定時適用**。
        #
        # 鎖定那一關必須看到真的標記,否則前一個目標留下的寬限會讓一個
        # 沒貼標記的人在 1.5 秒內被鎖定 —— 那就完全失去身分驗證的意義了。
        if locked_target is not None and self._last_ok is not None:
            if time.time() - self._last_ok < self.config.grace_seconds:
                return 1.0

        return 0.0

    def reset(self):
        self.last_result = EMPTY_RESULT
        self._last_ok = None


def draw_debug(image, result, color=(255, 0, 255)):
    """把偵測到的標記框畫在畫面上,給現場校正看。座標已是全畫面像素。"""
    if result is None or result.bbox_px is None:
        return
    x, y, w, h = result.bbox_px
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
