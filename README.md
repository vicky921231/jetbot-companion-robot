# 智慧伴護機器人　車上程式碼

東吳大學資料科學系畢業專題
**「基於邊緣運算與多感測器融合之智慧伴護機器人」**

本 repo 只涵蓋 **JetBot 車上**的跟隨與避障。車外的 MQTT 接收、異常判定與
LINE 推播由組員負責，不在此處。

平台：NVIDIA JetBot（Jetson Nano）+ JetPack 4.5
基礎：NVIDIA-AI-IOT/jetbot 官方 `notebooks/object_following/live_demo.ipynb`

---

## 從哪裡開始看

| 想知道什麼 | 看哪裡 |
|---|---|
| **整體設計、每支程式在做什麼** | [`程式碼說明.md`](程式碼說明.md) ← 主要文件 |
| 怎麼部署、怎麼接線、參數怎麼調 | [`object_following/README.md`](object_following/README.md) |
| 標記顏色怎麼校正 | [`object_following/marker_calibration/README.md`](object_following/marker_calibration/README.md) |
| 避障照片怎麼拍 | [`collision_avoidance/拍照蒐集清單.md`](collision_avoidance/拍照蒐集清單.md) |

---

## 一幀發生什麼事

```
① 相機 300×300
②   ↓ 縮成 224×168
    避障模型 ResNet18 → 被擋住的機率              ← 每幀都跑
③ 讀 VL53L0X 雷射距離（Arduino 推播的快取值，不阻塞）
④ 兩者用 OR 融合成「被擋住」布林
   │
   ├─ 被擋住 → sm.on_blocked() → 繞行／判定繞不過去 → 【早退】
   │
   └─ 沒擋住 → 人物偵測 SSD MobileNet
                 ↓ 篩出 label==1 的框
              sm.update() → 決定這一幀要做什麼
                 ↑ 過程中用 HSV 標記驗證身分
⑤ 執行：馬達差速 × 速度倍率、紅綠燈、蜂鳴器
⑥ 送一筆 JSON 出車（MQTT，有節流）
```

車上是**即時閉迴路**，每秒約 10 次，**不依賴網路**。斷網時車子照常運作。

---

## 設計原則

**決定「要做什麼」的程式碼，不准碰硬體。**

| 層 | 檔案 | 可以 import 什麼 |
|---|---|---|
| 決策 | `follow_state.py` | 只有標準函式庫 |
| 感知 | `marker.py` | cv2、numpy |
| 接線 | `live_demo_following_sm.ipynb` | 什麼都可以 |
| 輸出 | `feedback.py` | pyserial 或 Jetson.GPIO |
| 輸入 | `rangefinder.py` | 什麼都不用 |

好處有三個：狀態機能在筆電上完整驗證（42 組測試）、換硬體只動輸出層、
時間是餵進去的參數所以「連續 0.75 秒」能在毫秒內測完。

---

## 檔案

### `object_following/`　跟隨模組

| 檔案 | 在哪跑 | 職責 |
|---|---|---|
| `follow_state.py` | JetBot（可筆電測） | **狀態機**、跨幀追蹤、繞行、距離控制 |
| `marker.py` | JetBot（可筆電測） | 螢光雙色標記偵測（身分驗證） |
| `feedback.py` | JetBot | 紅綠燈／蜂鳴器輸出、序列埠、距離快取 |
| `rangefinder.py` | JetBot | 距離轉接頭 + 相機／雷射 OR 融合 |
| `live_demo_following_sm.ipynb` | JetBot | **接線層**，實際執行的入口 |
| `arduino_feedback/` | Arduino | 紅綠燈、蜂鳴器、VL53L0X 韌體 |
| `marker_calibration/` | 兩邊 | 標記顏色校正流程 |
| `sonar.py` | — | **已停用**，HC-SR04 時代的舊檔 |

### `collision_avoidance/`　避障模組

| 檔案 | 在哪跑 | 職責 |
|---|---|---|
| `data_collection_plus.ipynb` | JetBot | 拍訓練照片 |
| `train_collision.py` | 筆電 | 訓練二分類模型 |

---

## 測試

三支離線測試，筆電上跑，不需要 JetBot：

```bash
python test_follow_state.py    # 27 組　狀態機
python test_marker.py          #  5 組　標記偵測（合成影像）
python test_feedback.py        # 10 組　燈號對照與 Arduino 協定
```

實機互動測試（在 JetBot 上，需要有人看燈）：

```bash
python3 test_hardware.py
```

---

## 相對官方版本的差異

| 項目 | 官方 `live_demo.ipynb` | 本專題 |
|---|---|---|
| 選目標 | 每幀獨立選離中心最近的框 | 跨幀連續性追蹤 + 狀態機 |
| 開始跟隨 | 看到就跟 | 置中 + 夠近 + 0.75 秒 + 身分標記 |
| 身分驗證 | 無 | 螢光雙色標記 |
| 跟丟 | **盲目前進** | 保留 2 秒 + 方向性搜尋（只轉不前進） |
| 前進速度 | 固定 | 依距離縮放，太近停止但仍轉向 |
| 被擋住 | `robot.left(0.3)` | 繞行狀態：轉開 + 強制橫移 1.5 秒 |
| 繞不過去 | **無限空轉** | 三條件判定 + 停車 + 通知照護者 |
| 避障模型 | AlexNet | ResNet18 |
| 測距 | 無 | VL53L0X 雷射，與相機 OR 融合 |
| 使用者回饋 | 無 | 交通號誌 + 蜂鳴器 |
| 通報 | 無 | MQTT 送車外電腦 |
| 測試 | 無 | 42 組離線 + 1 支實機互動 |

---

## 不在版控裡的東西

| 東西 | 為什麼 | 怎麼取得 |
|---|---|---|
| `*.pth` 模型檔 | 單檔 43 MB，會隨重訓頻繁更換 | 走 OneDrive |
| `collision_avoidance/dataset/` | 含可辨識的人像 | 同上 |
| `marker_calibration/samples/` | 同上 | 同上 |
| `ssd_mobilenet_v2_coco.engine` | JetBot 映像檔本來就有 | 不需要 |

---

## 目前進度

| 項目 | 狀態 |
|---|---|
| 跟隨狀態機 | ✅ 軟體完成，42 組測試全過 |
| 標記身分驗證 | ⚠️ 程式完成，**HSV 門檻尚未實拍校正** |
| 紅綠燈／蜂鳴器／雷射 | ⚠️ 程式與韌體完成，**尚未接線驗證** |
| 避障模型 | ⚠️ 資料集 77 / 240 張，現有權重僅供煙霧測試 |
| **FPS** | ❌ **從未量測**。構想書目標 ≥ 15 |
| 車外 MQTT | ⏳ 等組員的 `sender.py` |
