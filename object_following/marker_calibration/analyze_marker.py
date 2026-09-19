"""標記顏色校正 —— 在筆電上跑,不需要 JetBot。

這支腳本取代了「訓練」這個步驟。HSV 門檻只有六個數字,但那六個數字必須
量出來,不能猜:同一張色紙在日光燈下、在窗邊、在走廊陰影裡,HSV 讀數會差很多。

兩個模式,對應兩件不同的事:

  --pick   量顏色。用**近距離特寫照**(約 0.3m),貼紙在畫面上夠大才框得準。
           框出桃紅區和第二色區,腳本統計實際的 HSV 分佈,印出建議門檻。

  --check  驗證距離。用 0.5 / 0.8 / 1.5 / 2.0m 的實拍照,套用目前 marker.py
           的門檻,報告每張圖抓到幾個像素、雙色在哪個距離開始失效。

用法(Windows PowerShell):
    python analyze_marker.py samples --pick
    python analyze_marker.py samples --check --save-masks

檔名規則:把距離寫進檔名就好,例如 `close_01.png`、`1.5m_03.png`。
腳本用正規式找 `數字m`,找不到也不會壞,只是報表少一欄。
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import marker  # noqa: E402


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

# 224x168 的圖直接顯示根本點不到 4x6 像素的標記,放大再讓人框。
# 用 INTER_NEAREST:放大時不要內插,否則會混出原本不存在的顏色。
DISPLAY_SCALE = 6


def list_images(folder):
    if not os.path.isdir(folder):
        sys.exit('找不到資料夾:%s' % folder)
    names = [n for n in sorted(os.listdir(folder))
             if n.lower().endswith(IMAGE_EXTS)]
    if not names:
        sys.exit('%s 裡面沒有圖片。先用 capture_samples.ipynb 拍幾張。' % folder)
    return [os.path.join(folder, n) for n in names]


def distance_from_name(path):
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*m', os.path.basename(path), re.I)
    return float(m.group(1)) if m else None


# --- 模式一:量顏色 --------------------------------------------------------

def pick_roi(image, title):
    """放大顯示讓人框一塊區域,回傳原始解析度下的 HSV 像素陣列。"""
    big = cv2.resize(image, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                     interpolation=cv2.INTER_NEAREST)
    box = cv2.selectROI(title, big, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    x, y, w, h = [int(round(v / float(DISPLAY_SCALE))) for v in box]
    if w <= 0 or h <= 0:
        return None
    roi = image[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    return cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)


def hue_ranges(hues):
    """回傳 [(lo, hi), ...]。色相是環狀的,分佈跨過 0/179 接縫時要拆成兩段。"""
    lo, hi = np.percentile(hues, [2, 98])
    if hi - lo <= 90:
        return [(int(max(0, lo - 3)), int(min(179, hi + 3)))]
    # 跨接縫:整體平移 90 度讓分佈連續,算完再移回來
    shifted = (hues.astype(np.int32) + 90) % 180
    slo, shi = np.percentile(shifted, [2, 98])
    lo2 = int((slo - 90 - 3) % 180)
    hi2 = int((shi - 90 + 3) % 180)
    return [(lo2, 179), (0, hi2)]


def summarise(name, pixels):
    """印出統計並回傳可直接貼進 marker.py 的門檻字串。"""
    h, s, v = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    clipped = np.count_nonzero((v >= 250) & (s < 60)) / float(len(pixels))

    print('')
    print('--- %s(%d 個像素)---' % (name, len(pixels)))
    print('  H  中位數 %3d   2%%~98%% %3d ~ %3d' % (
        np.median(h), np.percentile(h, 2), np.percentile(h, 98)))
    print('  S  中位數 %3d   5%%      %3d' % (np.median(s), np.percentile(s, 5)))
    print('  V  中位數 %3d   5%%      %3d' % (np.median(v), np.percentile(v, 5)))
    print('  過曝像素(V>=250 且 S<60):%.1f%%' % (clipped * 100))
    if clipped > 0.05:
        print('  >> 過曝偏高。這就是亮面材質的鏡面高光,那些像素的色相已經不可信。')
        print('     對策:改用霧面材質,或把相機的 exposure / gain 調低。')

    s_lo = int(max(0, np.percentile(s, 5) - 25))
    v_lo = int(max(0, np.percentile(v, 5) - 25))
    parts = ['((%d, %d, %d), (%d, %d, %d))' % (lo, s_lo, v_lo, hi, 255, 255)
             for lo, hi in hue_ranges(h)]
    return '[' + ', '.join(parts) + ']'


def run_pick(paths):
    print('')
    print('操作:滑鼠拖曳框出色塊 → 按 Enter 或空白鍵確認 → 按 c 跳過這一塊')
    print('建議用近距離特寫照來量顏色,框得準數字才準。')
    print('框「緊」一點,不要框到旁邊的褲子或地板。')

    pink_px, second_px = [], []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print('讀不到 %s,跳過' % path)
            continue
        print('')
        print('=== %s (%dx%d) ===' % (os.path.basename(path),
                                      image.shape[1], image.shape[0]))
        got = pick_roi(image, '1/2 framing PINK - %s' % os.path.basename(path))
        if got is not None:
            pink_px.append(got)
        got = pick_roi(image, '2/2 framing SECOND COLOUR - %s' % os.path.basename(path))
        if got is not None:
            second_px.append(got)

    if not pink_px:
        sys.exit('沒有框到任何桃紅像素,無法統計。')

    print('')
    print('=' * 68)
    print('統計結果')
    print('=' * 68)
    pink_line = summarise('桃紅(主鑑別色)', np.vstack(pink_px))
    second_line = None
    if second_px:
        second_line = summarise('第二色(確認色)', np.vstack(second_px))

    print('')
    print('=' * 68)
    print('把下面兩行貼進 marker.py 的 MarkerConfig,取代原本的估計值:')
    print('=' * 68)
    print('    pink_ranges = %s' % pink_line)
    if second_line:
        print('    second_ranges = %s' % second_line)
    print('')
    print('貼完再跑一次 --check 確認實拍照都抓得到。')


# --- 模式二:驗證距離 ------------------------------------------------------

def run_check(paths, save_masks):
    cfg = marker.MarkerConfig()
    out_dir = os.path.join(os.path.dirname(paths[0]), 'check_out')
    if save_masks and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print('')
    print('套用 marker.py 目前的門檻。bbox 用整張畫面,所以搜尋範圍是下方 %d%%。'
          % int(cfg.search_bottom_fraction * 100))
    print('')
    print('%-24s %7s %7s %9s %9s %7s' % (
        '檔名', '桃紅px', '第二色px', '只認桃紅', '雙色驗證', '過曝%'))
    print('-' * 68)

    rows = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        # 兩種判準都跑一次,才看得出雙色是在哪個距離開始失效
        loose = marker.detect(image, [0.0, 0.0, 1.0, 1.0], cfg, diagnostics=True)
        strict_cfg = marker.MarkerConfig()
        strict_cfg.dual_color_min_bbox_h = 0.0      # 強制要求雙色
        strict = marker.detect(image, [0.0, 0.0, 1.0, 1.0], strict_cfg)

        loose_cfg = marker.MarkerConfig()
        loose_cfg.dual_color_min_bbox_h = 2.0       # 強制只認桃紅
        pink_only = marker.detect(image, [0.0, 0.0, 1.0, 1.0], loose_cfg)

        name = os.path.basename(path)
        print('%-24s %7d %7d %9s %9s %6.1f%%' % (
            name[:24], strict.pink_area, strict.second_area,
            'OK' if pink_only.found else 'FAIL',
            'OK' if strict.found else 'FAIL',
            loose.clipped_ratio * 100))
        rows.append((distance_from_name(path), pink_only.found, strict.found))

        if save_masks:
            vis = image.copy()
            marker.draw_debug(vis, strict if strict.found else pink_only)
            big = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_dir, name), big)

    print('')
    known = [r for r in rows if r[0] is not None]
    if known:
        dual_ok = [d for d, _p, s in known if s]
        pink_ok = [d for d, p, _s in known if p]
        print('雙色驗證成功的最遠距離:%s' %
              ('%.2f m' % max(dual_ok) if dual_ok else '全部失敗'))
        print('只認桃紅成功的最遠距離:%s' %
              ('%.2f m' % max(pink_ok) if pink_ok else '全部失敗'))
        if dual_ok:
            print('')
            print('>> 把 marker.py 的 dual_color_min_bbox_h 設成「雙色最遠距離」')
            print('   對應的框高。框高 0.30 約 1.5m、0.45 約 1.0m、0.55 約 0.8m。')
    else:
        print('(檔名裡沒有距離資訊,跳過距離摘要。命名成 1.5m_01.png 就會有。)')

    if save_masks:
        print('')
        print('遮罩疊圖存到:%s' % out_dir)


def main():
    ap = argparse.ArgumentParser(description='螢光標記 HSV 門檻校正')
    ap.add_argument('folder', help='放實拍照片的資料夾')
    ap.add_argument('--pick', action='store_true', help='互動框選,量出 HSV 門檻')
    ap.add_argument('--check', action='store_true', help='套用目前門檻,驗證各距離')
    ap.add_argument('--save-masks', action='store_true', help='--check 時存疊圖')
    args = ap.parse_args()

    if not args.pick and not args.check:
        ap.error('至少要指定 --pick 或 --check')

    paths = list_images(args.folder)
    print('找到 %d 張圖片。' % len(paths))
    if args.pick:
        run_pick(paths)
    if args.check:
        run_check(paths, args.save_masks)


if __name__ == '__main__':
    main()
