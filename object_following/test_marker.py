"""marker.py 的離線測試。在筆電上直接跑,不需要 JetBot:

    python test_marker.py

用合成影像(灰底 + 兩個純色方塊)測邏輯,不是測門檻。
**真正的 HSV 門檻必須用實拍照校正**,見 marker_calibration/。

與 test_follow_state.py 分開的理由:那一支刻意不 import 任何第三方套件,
這一支需要 cv2 / numpy。
"""

import time

import numpy as np

import marker

PINK = (147, 20, 255)     # BGR,約 HSV H=164
YELLOW = (0, 255, 255)    # BGR,約 HSV H=30
FULL = [0.0, 0.0, 1.0, 1.0]


def blank():
    return np.full((168, 224, 3), 120, np.uint8)      # 中性灰背景


def with_marker(pink_rows=(120, 126), yellow=True, cols=(100, 106)):
    """在畫面下半部放一個「上桃紅 / 下螢光黃」的合成標記。"""
    img = blank()
    r0, r1 = pink_rows
    c0, c1 = cols
    img[r0:r1, c0:c1] = PINK
    if yellow:
        img[r1:r1 + (r1 - r0), c0:c1] = YELLOW
    return img


def check(name, cond):
    print(('  PASS  ' if cond else '  FAIL  ') + name)
    if not cond:
        raise AssertionError(name)


def test_detect_dual_colour():
    print('偵測:雙色標記')
    r = marker.detect(with_marker(), FULL, diagnostics=True)
    check('找到標記', r.found)
    check('桃紅面積正確(%d)' % r.pink_area, r.pink_area == 36)
    check('近距離會要求雙色', r.dual_required)
    check('抓到第二色(%d)' % r.second_area, r.second_area > 0)
    check('bbox_px 是全畫面座標 %s' % (r.bbox_px,),
          r.bbox_px is not None and r.bbox_px[0] == 100 and r.bbox_px[1] == 120)


def test_detect_negatives():
    print('偵測:不該誤判的情況')
    check('空白畫面', not marker.detect(blank(), FULL).found)
    check('近距離只有桃紅、缺第二色',
          not marker.detect(with_marker(yellow=False), FULL).found)
    # 標記出現在上半身 = 不在腿的位置,應被搜尋範圍排除
    check('標記在上半身', not marker.detect(with_marker(pink_rows=(20, 26)), FULL).found)
    # 整條腿都是桃紅 = 桃紅色褲子
    shirt = blank()
    shirt[90:168, 60:180] = PINK
    check('大片桃紅(衣物)被 max_pink_ratio 擋下', not marker.detect(shirt, FULL).found)


def test_distance_adaptive():
    print('距離自適應:遠距離退化成只認桃紅')
    cfg = marker.MarkerConfig()
    cfg.dual_color_min_bbox_h = 2.0          # 模擬遠距離(框高永遠達不到門檻)
    r = marker.detect(with_marker(yellow=False), FULL, cfg)
    check('遠距離只有桃紅也算找到', r.found)
    check('且不要求雙色', not r.dual_required)


def test_verifier_grace():
    print('遮擋寬限:只在已鎖定時適用')
    sentinel = object()                      # 假裝是 LockedTarget

    v = marker.MarkerVerifier()
    check('看到標記回 1.0', v(with_marker(), FULL, None) == 1.0)
    check('已鎖定 + 短暫遮住 → 寬限期內仍回 1.0',
          v(blank(), FULL, sentinel) == 1.0)

    v2 = marker.MarkerVerifier()
    v2(with_marker(), FULL, None)
    check('未鎖定 + 沒看到標記 → 0.0(寬限不適用於鎖定關卡)',
          v2(blank(), FULL, None) == 0.0)

    v3 = marker.MarkerVerifier()
    v3.config.grace_seconds = 0.05
    v3(with_marker(), FULL, None)
    time.sleep(0.08)
    check('已鎖定 + 超過寬限 → 0.0', v3(blank(), FULL, sentinel) == 0.0)


def test_speed():
    print('速度:必須遠小於一幀的預算')
    img = with_marker()
    t = time.time()
    for _ in range(500):
        marker.detect(img, FULL)
    per_ms = (time.time() - t) / 500 * 1000.0
    print('  每次 detect(): %.3f ms(筆電)' % per_ms)
    check('筆電上 < 2ms', per_ms < 2.0)


if __name__ == '__main__':
    tests = [
        test_detect_dual_colour,
        test_detect_negatives,
        test_distance_adaptive,
        test_verifier_grace,
        test_speed,
    ]
    for t in tests:
        t()
    print('\n全部通過(%d 組)' % len(tests))
