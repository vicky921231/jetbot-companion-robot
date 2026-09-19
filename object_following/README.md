# Object Following 跟隨鎖定狀態機

畢專「智慧伴護機器人」的跟隨模組。基礎是 NVIDIA-AI-IOT/jetbot 官方
`notebooks/object_following/live_demo.ipynb`，在其上加裝鎖定狀態機與跨幀連續性追蹤。

需求來源：`object_following_狀態機修改需求.md`

## 檔案

| 檔案 | 說明 |
|---|---|
| `follow_state.py` | 狀態機 + 跨幀追蹤。純標準函式庫，不 import jetbot / torch / cv2 / GPIO |
| `feedback.py` | 紅綠燈與蜂鳴器輸出、序列埠連線與距離快取、照護者通知的可抽換傳送介面 |
| `rangefinder.py` | VL53L0X 距離的轉接頭 + 相機／雷射的 OR 融合（`is_blocked()`） |
| `marker.py` | 螢光雙色標記偵測（HSV），接在狀態機的 `identity_score` 接縫上 |
| `live_demo_following_sm.ipynb` | 改寫版 notebook，實際在 JetBot 上執行的入口 |
| `arduino_feedback/arduino_feedback.ino` | Arduino 韌體：紅綠燈、蜂鳴器、VL53L0X 測距 |
| `marker_calibration/` | 標記顏色校正流程（JetBot 拍照 + 筆電分析），見該資料夾的 README |
| `test_follow_state.py` | 狀態機離線測試，27 組。筆電就能跑 |
| `test_marker.py` | 標記偵測離線測試，5 組。合成影像，不需要照片 |
| `test_feedback.py` | 燈號對照與 Arduino 協定的離線測試，10 組。不需要硬體 |
| `test_hardware.py` | 紅綠燈／蜂鳴器／測距實機測試，不啟動相機與馬達，桌面即可執行 |
| `sonar.py` | **已停用。** HC-SR04 時代的舊檔，被 `rangefinder.py` 取代 |
| `實作說明.md` | 改了什麼、為什麼這樣改、程式碼詳解。⚠️ 尚未同步 2026-09-18 改版 |
| `../參考_官方原始碼/live_demo.ipynb` | 官方未修改版，供對照用 |

決策邏輯（`follow_state.py`）與硬體 IO（`feedback.py`、notebook）刻意分開，
所以狀態機可以在筆電上完整驗證，不必佔用機器車。

三支離線測試在筆電上一次跑完：

```bash
python test_follow_state.py && python test_marker.py && python test_feedback.py
```

## 2026-09-18 改版摘要

| 原本 | 現在 | 為什麼 |
|---|---|---|
| 單顆 LED（D10） | **交通號誌模組**：紅 D10／黃 D9／綠 D8 | 單顆燈時 `waiting` 與 `impassable_wait` 都是慢閃、分不出來；三色語意對長輩也是零學習成本 |
| HC-SR04 超音波接 Jetson GPIO | **VL53L0X 雷射 ToF 接 Arduino I2C** | 不用分壓電阻、不怕布料吸音、不用在 Python 裡掐秒錶 |
| 序列埠 9600，問一句答一句 | **115200，Arduino 以 10Hz 主動推播** | 9600 下拿一次距離要 17ms，吃掉每幀 66ms 預算的四分之一 |
| re-ID 延到 v2 | **螢光雙色標記**（`marker.py`） | HSV 門檻約 1ms、零訓練資料；YOLO 要 30~50ms 且 Python 3.6 裝不了 |

⚠️ **鮑率變了，組員既有的測試腳本要跟著改一個數字。**

## 部署

`.py` 與 notebook 必須放在同一個資料夾，整包複製到 Jetson 的
`~/Notebooks/object_following/` 即可（該資料夾原本就有官方版與 `ssd_mobilenet_v2_coco.engine`）。

JetBot 上需要 `pip3 install pyserial`。Arduino IDE 需要安裝 **Adafruit VL53L0X** 程式庫。

修改過 `.py` 後要重啟 Jupyter kernel 才會生效。

## 離線測試

```
python test_follow_state.py
```

涵蓋需求文件的第 1、2、4、5、6、7 項，共 **24 組測試、120 項檢查**。時間是餵進去的參數而不是
`time.time()`，所以「連續 0.75 秒」這類條件在毫秒內就測完，整套跑完不到一秒。

## 狀態機

```
waiting ──目標置中且夠近──> confirming ──連續 0.75 秒──> locked
   ^                            |                          |
   |<────────目標離開───────────|                          |
   |<──────────────跟丟超過寬限時間────────────────────────|

任何狀態下被擋住 ──> detour(轉開 → 承諾橫移 1.5 秒) ──> 回到原狀態
                        │
                        └──判定繞不過去──> impassable_wait ──障礙排除──> 回到原狀態
```

`detour` 分兩階段：**turn**（原地左轉直到不再被擋住）→ **advance**（強制直行 1.5 秒，
期間完全不理會目標的拉力）。第二階段是必要的，否則轉開的下一幀就會被跟隨邏輯拉回
原方向，見下方「已知限制」。

### 跟丟後的方向性搜尋

目標在 `locked` 狀態下從畫面消失時，不會立刻停住。系統會看**消失前的水平偏移**：

- 偏移超過 `search_min_offset` → 朝那一側**原地旋轉**最多 `search_turn_seconds`，試著把鏡頭追過去
- 偏移太小（從正中央消失，多半是偵測漏幀或有人經過）→ 不轉，原地等

這是為直角牆角設計的：目標轉彎時會先飄到畫面邊緣、再被牆擋住，方向是明確的線索，
立刻停住等於白白丟掉它。轉到看見目標就立刻接回跟隨，不需要重新站定。

**搜尋只旋轉、不前進**——看不到人還往前開是官方版最危險的行為，需求文件項目 5 已明訂移除。
搜尋也**不會延長** `lost_grace_seconds`，超過寬限一樣解除鎖定。

### 判定繞不過去

「繞不過去」有三個判定條件，任一成立即停下並通知照護者：

- **(a) 連續被擋超過 5 秒** —— 正對牆壁、樓梯這類轉不開的情況
- **(b) 8 秒滑動視窗內被擋比例超過一半** —— 擺盪偵測
- **(c) 連續繞行 3 次仍過不去** —— 障礙物比橫移距離寬

| 階段 | 紅綠燈 | 蜂鳴器 |
|---|---|---|
| waiting | 🔴 慢閃 | 無聲 |
| confirming | 🟡 快閃 | 無聲 |
| locked（含跟丟搜尋中） | 🟢 恆亮 | **短鳴一聲**（進入時） |
| detour | 🟢 恆亮 ＋ 🟡 慢閃 | 無聲 |
| impassable_wait | 🔴 恆亮 | 無聲 |

`detour` 的綠燈刻意與 `locked` 相同——繞行期間鎖定並沒有解除，
對目標者而言「還跟著你」的訊息不該改變；黃燈是額外疊上去的資訊，不推翻原本的設計決定。

`waiting`（紅慢閃）與 `impassable_wait`（紅恆亮）終於分得開了。
單顆 LED 時兩者都是慢閃、長得一模一樣，但這兩個狀態意義天差地遠：
一個是「等你站好」，一個是「我卡住了，已經通知照護者」。

蜂鳴器只在「鎖定完成」那一刻使用。跟丟重新鎖定、無法跨越障礙一律不對目標者發出警示音
——不應把系統的問題轉嫁成要求目標者做出反應的負擔。

對照表的程式碼在 `feedback.py` 的 `STATE_LIGHTS`，`test_feedback.py` 會驗證它。

## 接線（全部接在 Arduino 上）

Jetson 與 Arduino 之間只有一條 USB 線。**Jetson 的 40-pin GPIO 完全不用碰。**

### 交通號誌模組（共陰，4 pin）

| 模組 | Arduino | 備註 |
|---|---|---|
| GND | GND | |
| R | **D10** | 沿用前一版已實測的腳位，高電位亮 |
| Y | **D9** | |
| G | **D8** | |

> ⚠️ **先確認模組上有沒有限流電阻。** 看不出來就每一路串 **220Ω**——直接接會超過
> Arduino 單腳 20mA 上限，久了會燒腳位。串了只是稍微暗一點，沒有壞處。

若買到的是**共陽**模組（低電位亮），把 `feedback.py` 的 `ACTIVE_HIGH` 改成 `False`，
接線不用動。`test_hardware.py` 的步驟 1 會告訴你是哪一種。

### 有源蜂鳴器

| 蜂鳴器 | Arduino |
|---|---|
| ＋ | **D11** |
| － | GND |

若是**無源**蜂鳴器（固定高電位不會響，需要 PWM 方波），
把 `.ino` 的 `ACTIVE_BUZZER` 改成 `false` 再燒錄一次。

### VL53L0X 雷射測距（GY-530）

| 模組 | Arduino Uno | 備註 |
|---|---|---|
| VIN | 5V | 板上有穩壓，3~5V 都可以 |
| GND | GND | |
| SCL | **A5** | I2C 腳位固定，不能改 |
| SDA | **A4** | |

Arduino IDE 需安裝 **Adafruit VL53L0X** 程式庫（工具 → 管理程式庫）。

**相對 HC-SR04 的三個改善：**

1. **不用分壓電阻。** 舊方案的 ECHO 輸出 5V、Jetson GPIO 只耐 3.3V，接錯會燒板子。
   現在測距完全在 Arduino 這邊，而且走 I2C，那個地雷整個消失
2. **不用在 Python 裡掐秒錶。** 舊方案要「送脈衝 → 等回波 → 量時間」，而 Linux 不是
   即時作業系統，那個迴圈隨時會被排程打斷。現在測距與計時都在感測器晶片內部完成
3. **不怕布料。** 超音波遇到衣服、毛衣、海綿會吸音——對「跟著一個長輩」特別不利。
   雷射不管表面軟硬，只要會反光就量得到

**代價**（要誠實寫進報告）：量程 2m（HC-SR04 是 4m，但這裡只需要 0.8m 內）；
**深色與黑色表面會吸收紅外線**，讀數會變短或直接失敗；
視野角約 25 度，比 HC-SR04 的 15 度還寬，正前方以外的東西更容易混進讀數裡。

### 安裝位置

**架設於與紅綠燈相同的高度，朝車體正前方。** 它的角色是**測距**，
不是「補相機看不到的東西」——它提供碰撞模型給不了的東西：一個實際的距離數值。
兩者性質不同，所以融合用 OR 而不是加權平均（見 `rangefinder.is_blocked()`）。

### 感測器沒接或故障時的行為

不用任何開關保護。HC-SR04 時代的風險是「未接線的 GPIO 浮接會讀到隨機電位、
回報假的近距離」，而 I2C 沒接就是沒回應：韌體開機會印 `READY vl53l0x=absent`，
不推播距離，`RangeFinder.distance` 永遠回 `None`。

**`None` 一律當成「不知道」而非「前方淨空」**，融合邏輯單獨採信相機判斷，
感測器故障時不會因此放行。所以可以**先燒韌體、先測紅綠燈，之後再接感測器**。

### 通訊協定（115200 8N1）

Jetson → Arduino：`LIGHT:RYG`（例 `LIGHT:100` = 只亮紅燈）、`LED_ON` / `LED_OFF`
（向後相容，映射到紅燈）、`BEEP` / `BEEP:<ms>`、`PING`、`DIST`

Arduino → Jetson：`READY vl53l0x=ok|absent`、`D:<mm>`（約每 100ms 一筆，
`-1` 代表超出量程或讀取失敗）、以及各指令的回覆

**序列埠只有一個擁有者**：`feedback.ArduinoBackend`。它開一條背景執行緒把每一行讀掉，
`D:` 存成距離並做中位數濾波，其餘丟棄。`rangefinder.py` 只向它要數字，不另外開連線。

這條讀取執行緒是必要的：Arduino 以 10Hz 主動推播，沒人讀的話 Linux 端的接收緩衝區會塞滿，
之後讀到的都是過期資料。而改成「問一句答一句」的話，camera callback 每幀都要等回覆。

## 現場實測前必須確認的三件事

1. **韌體已燒錄、鮑率對得上**：`feedback.py` 的 `ARDUINO_BAUD = 115200` 必須與
   `arduino_feedback.ino` 的 `Serial.begin()` 一致，否則序列埠全是亂碼。
   埠位址 `ARDUINO_PORT = '/dev/ttyACM0'`，用 `ls /dev/ttyACM* /dev/ttyUSB*` 確認。
   JetBot 上需先 `pip3 install pyserial`。
   **先跑 `python3 test_hardware.py`**——它會逐顆點燈、驗證極性、確認測距有讀數。
2. **`label_widget` 是否正確框到人**：官方 notebook 說明文字寫 "Person (index 0)"
   但程式碼預設 `value=1`，兩者不一致。此處以程式碼為準（COCO label map 中 person 為 1，
   0 保留給 background），但務必實測確認。
3. **`sender.py` 是否就位**：車外通報走 MQTT，由組員提供 `sender.py`，
   放在同一資料夾即可。notebook 會自動 `import sender`，失敗就退回 console 印字，
   不影響跟隨。JetBot 上需要 `pip3 install paho-mqtt`。

## 車外通報（MQTT）

架構定案：**車上只送原始狀態，判斷異常與推播 LINE 都在車外電腦。**
車上不直接推 LINE，否則照護者會收到重複訊息。

送出的格式（與組員的約定）：

```python
{
  "state":      "locked",              # 五種狀態之一
  "events":     ["lock_acquired"],     # 這一幀發生的事件，可能是空陣列
  "has_target": True,
  "readings": {
    "blocked_prob": 0.12,              # 碰撞模型機率
    "distance_m":   0.90,              # 超音波距離，未啟用時為 null
    "speed_scale":  1.0,               # 速度倍率，0 代表因為太近而停住
    "target_h":     0.42               # 目標框高，無目標時為 null
  }
}
```

`events` 是**陣列**，因為同一幀可能同時發生多件事。

**節流在車上做**：狀態改變或有事件時立刻送，其餘每秒最多一次，
所以 `sender.py` 只需負責連線與發送。送出失敗只印一次警告，
**不會中斷跟隨**——車子的安全不該取決於 Wi-Fi 通不通。

## 待現場實測校正的參數

全部集中在 notebook 的「載入狀態機與回饋硬體」那一格，改那裡即可。

| 參數 | 暫定值 | 如何量測 |
|---|---|---|
| `min_bbox_height` | 0.30 | 攝影機安裝角度定案後，實拍 0.5m / 1m / 1.5m 站立距離。notebook 會把目標框高比例 `h=0.xx` 印在畫面左上角，直接讀 |
| `track_center_max_dist` | 0.25 | 用錄影檔離線調 |
| `track_height_ratio_max` | 1.6 | 用錄影檔離線調 |
| `impassable_after_blocked_seconds` | 5.0 | 實測走廊轉角要幾秒才轉得開，設太短會把轉角誤判成死路 |
| `block_window_seconds` | 8.0 | 擺盪偵測的滑動視窗。判定需累積滿一整個視窗，所以最壞情況會先擺盪約 8 秒才停下 |
| `block_ratio_threshold` | 0.5 | 視窗內被擋比例門檻。調高較不易誤判但更慢察覺 |
| `search_turn_seconds` | 1.5 | 跟丟後朝最後方向轉多久。太短轉不到，太長會轉過頭 |
| `search_min_offset` | 0.12 | 消失前的水平偏移要超過此值才轉。調低容易被偵測抖動誤導 |
| `SEARCH_TURN_SPEED` | 0.25 | 搜尋時的旋轉速度（notebook 內）。比避障的 0.3 慢，避免轉過頭 |
| `detour_advance_seconds` | 1.5 | 繞行橫移時間。太短繞不過去，太長會偏離目標太遠。用實際會遇到的障礙物寬度試 |
| `max_detour_attempts` | 3 | 連續繞這麼多次仍過不去就放棄並通知 |
| `RANGE_BLOCK_DISTANCE` | 0.35 | 雷射判定被擋的距離（公尺）。依車體長度與煞停距離實測 |
| `speed_widget` / `turn_gain_widget` | 0.4 / 0.8 | 需分別針對孩童與長者的移動速度校正，滑桿保留供現場調整 |

標記偵測的 HSV 門檻（`marker.py` 的 `MarkerConfig`）**不在這張表裡**——
它們不能用「試出來」的方式調，要用 `marker_calibration/` 的流程量出來。

建議**一次錄影解決全部**：錄一段「目標者走動 + 另一人穿不同顏色衣服入鏡 + 各種光線位置」，
之後離線調參數，不要為了不同參數跑好幾趟。同一份影片日後加 re-ID 時也用得上。

## 已知限制

**繞行是開環的，不保證成功。** 官方的 `robot.left(0.3)` 是左右輪反轉的**原地自轉**，
車體沒有任何橫向位移，只有車頭朝向改變。若轉開後立刻讓跟隨邏輯把車頭拉回目標，
就會直接正對回障礙物，形成：

```
左轉到不擋住 → 跟隨邏輯把車頭拉回目標 → 又正對石頭 → 再左轉 → ……
```

的來回擺盪。本版用 `detour` 狀態修掉這件事：轉開之後**強制直行 1.5 秒**才交還給跟隨
邏輯，讓車體真的產生橫向位移。**修掉擺盪的是這個「承諾」，不是超音波本身**——超音波
提供的是距離數值，讓「被擋住」能用絕對距離門檻判定，與擺盪無關。

但這個繞行是**開環**的：橫移的時間是固定值，系統不知道自己實際偏開了多少，也不知道
障礙物有多寬。石頭小就過得去，障礙寬一點就過不去。因此保留了三重放棄條件——連續被擋
5 秒、8 秒視窗內被擋過半、連續繞行 3 次仍過不去——任一成立就停下來通知照護者。
機器人不會假裝自己能通過，也不會靜悄悄地卡死。

**要做到閉環繞行，需要方向性的距離資訊。** 目前是單顆前向超音波，只知道「前面多遠有東西」，
不知道障礙在左邊還右邊，所以繞行方向固定往左（沿用官方行為）。加到**左/中/右三顆**
就能選擇往空曠的那側繞、並在橫移時用側向感測器確認障礙物已經通過，才算真正的閉環。
`sonar.py` 的 `Sonar` 是單顆封裝，多顆時各自建立實例即可（注意要**輪流觸發**避免串音，
會使有效取樣率下降為 1/N）。

**跟丟後可能跟錯人。** 當原目標被遮擋或偵測漏框，而畫面中還有其他人時，系統會退而選
「離畫面中心最近」的候選框繼續跟隨，過程中不做任何身分驗證。因此若目標被擋住的瞬間
剛好有路人經過，機器人可能跟著路人離開，且無法自我察覺。

這是本版本刻意接受的取捨：完整解法需要 re-ID（外觀特徵比對），而深度 re-ID 模型在
Jetson Nano 4GB 上與現有的兩個模型（AlexNet 碰撞避障 + SSD MobileNet 偵測）競爭
記憶體與算力，成本不合畢專時程。

**下一版的作法已預留接縫**，`follow_state.py` 中：

```python
sm.signature_fn   = lambda image, bbox: 抽取特徵(image, bbox)
sm.identity_score = lambda image, bbox, locked: 比對相似度(...)
config.identity_threshold = 實測門檻
```

`image` 參數從第一版就串進目標選擇路徑，所以加 re-ID 時**不需要改動狀態機本身**。
`test_follow_state.py` 的 `test_reid_seam_rejects_stranger` 已驗證：換上會回傳低分的
`identity_score` 後，跟丟時的行為自動變成「寧可原地等待，也不跟錯人」。

第一版 re-ID 建議做 HSV 顏色直方圖（取框高 20%~55% 的上半身區域，避開臉與腳），
運算成本約 1~2ms，不載入任何模型；深度 re-ID（如 OSNet）列為未來工作。

## 不在本模組範圍

跌倒偵測（MPU-6050 IMU 雙條件判斷）另開獨立模組，不與此處的 camera callback 合併。
