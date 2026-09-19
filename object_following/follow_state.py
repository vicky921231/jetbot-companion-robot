"""JetBot 伴護機器人:跟隨鎖定狀態機(純邏輯層)。

這支檔案刻意不 import jetbot / torch / cv2 / Jetson.GPIO,只用標準函式庫。
理由有兩個:

1. 可以在筆電上直接跑單元測試(見 test_follow_state.py),不必佔用機器車
2. execute() callback 只負責「取影像 → 問狀態機 → 執行動作」,IO 與決策分離

對應需求文件「落差分析」表格的第 1、2、4、5、6、7 項。
"""

import math
from collections import deque, namedtuple


# --- 狀態 -----------------------------------------------------------------

STATE_WAITING = 'waiting'            # 等待目標站定(含畫面中完全沒人)
STATE_CONFIRMING = 'confirming'      # 條件成立,倒數中
STATE_LOCKED = 'locked'              # 已鎖定,跟隨中
STATE_DETOUR = 'detour'              # 繞行中(先轉開,再承諾橫移一段)
STATE_IMPASSABLE = 'impassable_wait'  # 判定繞不過去,暫停跟隨

# 繞行的兩個階段
DETOUR_TURN = 'turn'        # 原地左轉,直到不再被擋住
DETOUR_ADVANCE = 'advance'  # 直行一段,讓車體真的產生橫向位移

# --- 事件(由呼叫端決定要接 LED / 蜂鳴器 / LINE 的哪一條)------------------

EVENT_ENTER_CONFIRMING = 'enter_confirming'
EVENT_CANCEL_CONFIRMING = 'cancel_confirming'
EVENT_LOCK_ACQUIRED = 'lock_acquired'      # ← 唯一該響蜂鳴器的事件
EVENT_LOCK_LOST = 'lock_lost'
EVENT_RELOCKED = 'relocked'                # 項目4:跟丟後退而求其次重新鎖定
EVENT_SEARCHING = 'searching'              # 跟丟後朝最後已知方向轉,嘗試找回
EVENT_NOTIFY_NO_TARGET = 'notify_no_target'
EVENT_ENTER_DETOUR = 'enter_detour'
EVENT_DETOUR_ADVANCE = 'detour_advance'
EVENT_DETOUR_DONE = 'detour_done'
EVENT_ENTER_IMPASSABLE = 'enter_impassable'
EVENT_NOTIFY_IMPASSABLE = 'notify_impassable'
EVENT_RESUME_FOLLOW = 'resume_follow'

# --- 馬達動作 -------------------------------------------------------------

MOTOR_STOP = 'stop'
MOTOR_STEER = 'steer'         # 依 action.steer 做差速
MOTOR_TURN_LEFT = 'turn_left'  # 碰撞避障:官方的原地左轉
MOTOR_TURN_RIGHT = 'turn_right'  # 跟丟後朝最後已知方向找回目標
MOTOR_FORWARD = 'forward'      # 繞行的橫移階段:直行,暫時不理會目標的拉力

Action = namedtuple('Action',
                    ['state', 'motor', 'steer', 'target', 'events', 'speed_scale'])


# --- bbox 小工具(bbox 為正規化座標 [x1, y1, x2, y2],值域 0~1)-----------

def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_height(bbox):
    return abs(bbox[3] - bbox[1])


def center_offset(bbox):
    """回傳 x 方向偏離畫面中心的量,值域約 -0.5 ~ +0.5(沿用官方 detection_center 的慣例)。"""
    return bbox_center(bbox)[0] - 0.5


def _ramp(value, at_zero, at_one):
    """線性斜坡:value 在 at_zero 回 0、在 at_one 回 1,中間內插,兩端夾住。

    at_zero 可以大於 at_one(框高就是這種情況:框越大代表越近、越該慢),
    所以這個函式對兩個方向都適用。
    """
    span = at_one - at_zero
    if span == 0:
        return 1.0
    return max(0.0, min(1.0, (value - at_zero) / span))


def _dist_from_frame_center(bbox):
    cx, cy = bbox_center(bbox)
    return math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)


class FollowConfig(object):
    """所有可調參數集中在這裡,方便現場實測時只改一個地方。"""

    # --- 已定案(需求文件「已經可以直接當規格用的參數」)---
    center_tolerance = 0.15      # 鎖定所需的水平置中容許誤差:畫面寬 ±15%
    confirm_seconds = 0.75       # 鎖定所需連續持續時間,規格為 0.5~1.0 秒

    # --- 待現場實測定案 ---
    # TODO: 攝影機安裝角度確定後,實拍 0.5m / 1m / 1.5m 站立距離量測框高比例再定案
    min_bbox_height = 0.30       # 候選框最小高度比例(距離門檻)

    # --- 跨幀連續性追蹤(項目1)的幾何門檻,建議用錄影檔離線調 ---
    track_center_max_dist = 0.25   # 與上一幀鎖定框的中心距離上限
    track_height_ratio_max = 1.6   # 與上一幀鎖定框的高度比值上限(取大/小)

    # --- 跟隨距離控制 ---
    # 官方版的前進速度是固定的,完全不看目標多遠,所以人停下來時機器人會一路
    # 逼到碰撞模型判定被擋住為止。這裡改成依距離縮放速度:接近到 follow 就開始
    # 減速,到 stop 就完全停住(但仍會轉向,保持鏡頭對著人)。
    #
    # 距離有兩個來源,優先用超音波,沒有才退回框高:
    #   超音波  量到的是絕對距離,準,但它不知道量到的是誰
    #   框高    一定是目標本人,但要校正,且受身高與姿勢影響
    #
    # TODO: 兩組門檻都要現場實測。
    follow_distance_m = 0.80     # 超音波:近於此開始減速
    stop_distance_m = 0.45       # 超音波:近於此完全停止
    slow_bbox_height = 0.55      # 框高:大於此開始減速(框越大代表越近)
    stop_bbox_height = 0.75      # 框高:大於此完全停止

    # 超音波只在目標大致位於正前方時才可信——偏到側邊時,波束打到的
    # 是牆或別的東西,不是被跟隨的人。
    distance_trust_offset = 0.15

    # --- 跟丟寬限(項目5、6)---
    lost_grace_seconds = 2.0     # 畫面中完全沒人時,鎖定還保留多久才真正放棄

    # --- 跟丟後的方向性搜尋 ---
    # 直角牆角的情境:目標轉彎時會先飄到畫面邊緣,然後被牆擋住而突然消失。
    # 這時「立刻停住」等於放棄了最有用的線索——我們明明知道他往哪邊去。
    # 所以朝最後已知的偏移方向原地轉一小段,讓鏡頭追過去。
    #
    # 只有「消失前確實偏向某一側」才轉。若是從正中央消失(偵測漏幀、
    # 有人從前面經過),方向資訊不可靠,轉了反而會把鏡頭轉離目標。
    #
    # 刻意只原地旋轉、不前進:看不到人還往前開是官方版最危險的行為,
    # 需求文件項目5 已明訂要移除。原地轉不會離開原位,不違反這條。
    search_turn_seconds = 1.5    # 最多朝那個方向轉多久(不會超過跟丟寬限)
    search_min_offset = 0.12     # 消失前的水平偏移要超過這個量才值得轉

    # --- 無法跨越的障礙(項目7)---
    # 碰撞避障模型只回答「擋住/沒擋住」,不必也不需要分辨樓梯——JetBot 對樓梯和
    # 牆角一樣過不去。因此改用「被擋住的時間」來區分繞得開與繞不開,分兩種判定:
    #
    # (a) 連續被擋超過 impassable_after_blocked_seconds → 直接判定繞不過去
    # (b) 滑動視窗內被擋的比例過高 → 判定為「擺盪」,同樣繞不過去
    #
    # (b) 是必要的:官方的 robot.left() 是原地自轉,車體沒有橫向位移。
    # 障礙物擋在機器人與目標之間時,會出現「左轉到不擋住 → 跟隨邏輯把車頭拉回目標
    # → 又正對障礙物」的來回擺盪。這種情況下每次被擋的持續時間都很短,
    # (a) 永遠不會觸發,機器人會卡在原地無限擺盪且不通知任何人。
    impassable_after_blocked_seconds = 5.0   # (a) 連續被擋門檻
    block_window_seconds = 8.0               # (b) 滑動視窗長度
    block_ratio_threshold = 0.5              # (b) 視窗內被擋比例超過此值即判定擺盪

    # --- 繞行(純時間邏輯,不依賴任何額外感測器)---
    # 擺盪的根因不是「不知道有障礙」,而是轉開之後沒有任何承諾,立刻被跟隨邏輯
    # 拉回原方向。所以轉開後強制直行一段時間,讓車體真的產生橫向位移,
    # 這段期間完全不理會目標的拉力。這才是修掉擺盪的關鍵,不是感測器本身。
    # TODO: 現場實測。太短繞不過去,太長會偏離目標太遠。
    detour_advance_seconds = 1.5
    max_detour_attempts = 3        # 連續繞行這麼多次仍過不去 → 判定繞不過去
    detour_reset_seconds = 10.0    # 順利跟隨這麼久,繞行次數歸零

    # --- 身分驗證(re-ID)的接縫,v1 不啟用 ---
    # v1: identity_score() 永遠回 1.0、門檻 0.0 → 一定通過。
    # v2 加上顏色直方圖或深度 re-ID 後,只要換掉 identity_score 並把門檻調上來,
    #    狀態機其餘部分完全不用動。
    identity_threshold = 0.0

    # --- 通知節流(避免每幀都打 API)---
    no_target_notify_delay = 3.0   # 連續幾秒找不到人才通知照護者
    notify_cooldown = 60.0         # 同一種通知的最短重發間隔


class LockedTarget(object):
    """目前鎖定的目標。刻意用物件而不是裸 bbox,好讓 signature 之後能直接塞進來。"""

    def __init__(self, bbox, signature, now):
        self.bbox = bbox
        self.signature = signature   # v1 為 None;v2 存顏色直方圖 / re-ID embedding
        self.locked_at = now
        self.last_seen = now

    def update(self, bbox, now):
        self.bbox = bbox
        self.last_seen = now


def default_identity_score(image, bbox, locked_target):
    """v1 佔位:永遠通過。

    v2 要加 re-ID 時,把這個函式換成真正的比對(回傳 0~1 的相似度),
    並把 FollowConfig.identity_threshold 調到實測出來的門檻即可。
    注意 image 參數從第一天就串進來了,就是為了這一刻不用回頭改介面。
    """
    return 1.0


def default_signature_fn(image, bbox):
    """v1 佔位:不抽任何特徵。v2 換成顏色直方圖 / embedding 抽取。"""
    return None


class FollowStateMachine(object):

    def __init__(self, config=None):
        self.config = config or FollowConfig()
        # 可替換的接縫:sm.identity_score = my_func 即可換掉,不用改狀態機
        self.identity_score = default_identity_score
        self.signature_fn = default_signature_fn
        self.reset()

    def reset(self):
        self.state = STATE_WAITING
        self._locked = None
        self._confirm_started = None
        self._confirm_bbox = None
        self._absent_since = None
        self._last_no_target_notify = None
        self._last_impassable_notify = None
        self._blocked_since = None
        self._block_history = deque()   # [(時間, 是否被擋住)],用於擺盪偵測
        self._detour_phase = None
        self._detour_advance_started = None
        self._detour_attempts = 0
        self._last_detour_end = None
        self._search_started = None
        self._search_direction = 0   # -1 往左,+1 往右,0 不轉

    # -- 碰撞避障期間呼叫,取代官方直接寫死的 robot.left(0.3)(項目6、7)-----
    def on_blocked(self, now):
        """碰撞模型判定被擋住時呼叫,回傳這一幀該做的動作。

        做兩件事:

        1. 凍結跟丟計時,讓避障結束後能銜接回上一次的鎖定框,而不是因為寬限
           時間被吃光而被迫要求目標者重新站定一次(項目6)
        2. 被擋住太久時升級成 impassable_wait(項目7)——繞不開就當作跨不過去,
           停下來通知照護者。連續被擋、以及左右來回擺盪,兩種都算
        """
        events = []
        if self._locked is not None:
            self._locked.last_seen = now
        self._absent_since = None
        # 改由避障接管,先前的搜尋方向已經沒有意義
        self._search_started = None
        self._search_direction = 0
        self._record_block_sample(now, True)

        if self._blocked_since is None:
            self._blocked_since = now

        if now - self._blocked_since >= self.config.impassable_after_blocked_seconds:
            return self._handle_impassable(now, events)
        if self._is_thrashing(now):
            return self._handle_impassable(now, events)

        if self.state != STATE_DETOUR:
            action = self._enter_detour(now, events)
            if action is not None:      # 繞行次數用盡 → 判定繞不過去
                return action
        else:
            # 橫移到一半又被擋住 → 退回轉向階段,重新找出路
            self._detour_phase = DETOUR_TURN

        return self._act(MOTOR_TURN_LEFT, 0.0, None, events)

    # -- 繞行 ---------------------------------------------------------------
    def _enter_detour(self, now, events):
        """回傳非 None 代表已升級為 impassable,呼叫端應直接回傳它。"""
        cfg = self.config
        if (self._last_detour_end is None
                or (now - self._last_detour_end) > cfg.detour_reset_seconds):
            self._detour_attempts = 1
        else:
            self._detour_attempts += 1

        if self._detour_attempts > cfg.max_detour_attempts:
            return self._handle_impassable(now, events)

        self.state = STATE_DETOUR
        self._detour_phase = DETOUR_TURN
        events.append(EVENT_ENTER_DETOUR)
        return None

    def _continue_detour(self, now, events):
        """在沒被擋住的幀推進繞行。回傳 None 代表繞行結束,交還給跟隨邏輯。"""
        cfg = self.config

        if self._detour_phase == DETOUR_TURN:
            # 這一幀沒被擋住 = 已經轉開了,開始橫移
            self._detour_phase = DETOUR_ADVANCE
            self._detour_advance_started = now
            events.append(EVENT_DETOUR_ADVANCE)

        if now - self._detour_advance_started < cfg.detour_advance_seconds:
            # 橫移期間凍結跟丟計時:目標可能暫時不在畫面中央甚至出框,
            # 但這是我們自己轉開造成的,不該因此判定跟丟。
            if self._locked is not None:
                self._locked.last_seen = now
            self._absent_since = None
            return self._act(MOTOR_FORWARD, 0.0, None, events)

        self.state = STATE_LOCKED if self._locked is not None else STATE_WAITING
        self._detour_phase = None
        self._detour_advance_started = None
        self._last_detour_end = now
        events.append(EVENT_DETOUR_DONE)
        return None

    # -- 擺盪偵測 -----------------------------------------------------------
    def _record_block_sample(self, now, blocked):
        cfg = self.config
        self._block_history.append((now, blocked))
        cutoff = now - cfg.block_window_seconds
        while self._block_history and self._block_history[0][0] < cutoff:
            self._block_history.popleft()

    def _is_thrashing(self, now):
        """視窗內被擋比例過高 → 左轉繞不開,只是在原地來回。"""
        cfg = self.config
        history = self._block_history
        if len(history) < 2:
            return False
        # 必須累積滿一整個視窗才下判斷。用半個視窗會誤判:合法繞行 2 秒後恢復跟隨,
        # 在視窗只有半滿時被擋比例仍會達到 50%,好端端繞開的轉角會被當成死路。
        if history[-1][0] - history[0][0] < cfg.block_window_seconds:
            return False
        blocked = sum(1 for _, is_blocked in history if is_blocked)
        return (float(blocked) / len(history)) >= cfg.block_ratio_threshold

    # -- 主要進入點 ---------------------------------------------------------
    def update(self, candidates, image, now, impassable=False, distance_m=None):
        """candidates: 已依 label 篩選過的偵測框 list,每個是 {'label':.., 'bbox': [...]}
        image:       當幀原始影像,v1 不使用,保留給 re-ID
        now:         time.time()
        impassable:  額外的無法跨越判定來源。留著給日後真的加測距/懸崖感測器用;
                     v1 不需要——項目7 由 on_blocked() 依被擋住的持續時間自行升級。
        """
        events = []

        # 能走到 update() 就代表這一幀沒被擋住,連續計時歸零
        self._blocked_since = None
        self._record_block_sample(now, False)

        if impassable:
            return self._handle_impassable(now, events)

        # 擺盪偵測要在沒被擋住的幀也做:左右來回時,一半的幀本來就是「沒擋住」,
        # 只在 on_blocked() 裡檢查會漏掉一半的判定時機。
        if self._is_thrashing(now):
            return self._handle_impassable(now, events)

        if self.state == STATE_IMPASSABLE:
            self.state = STATE_LOCKED if self._locked is not None else STATE_WAITING
            events.append(EVENT_RESUME_FOLLOW)

        # 繞行中:先把承諾的橫移走完,再讓跟隨邏輯接手。
        # 少了這一段,轉開的下一幀就會被跟隨邏輯拉回原方向,又正對障礙物。
        if self.state == STATE_DETOUR:
            action = self._continue_detour(now, events)
            if action is not None:
                return action

        self._track_absence(candidates, now, events)

        det, matched_previous = self.select_target(candidates, image)

        if self.state == STATE_LOCKED:
            return self._update_locked(det, matched_previous, now, events, distance_m)
        return self._update_unlocked(det, image, now, events)

    # -- 項目1 + 項目4:目標選擇 -------------------------------------------
    def select_target(self, candidates, image):
        """回傳 (detection, matched_previous)。

        matched_previous 為 True 代表這個框是跟上一幀鎖定框「對得起來」的延續;
        False 代表沒對上,是退而求其次選的最近候選框(項目4 的已知風險)。
        """
        if not candidates:
            return None, False

        cfg = self.config
        locked = self._locked

        if locked is not None:
            best = None
            best_cost = None
            for det in candidates:
                cost = self._continuity_cost(det['bbox'], locked.bbox)
                if cost is None:
                    continue
                if self.identity_score(image, det['bbox'], locked) < cfg.identity_threshold:
                    continue
                if best_cost is None or cost < best_cost:
                    best, best_cost = det, cost
            if best is not None:
                return best, True

        # 項目4:對不上任何延續框 → 退而選畫面中離中心最近的候選框。
        # 已知風險:這裡沒有身分驗證,可能跟到路過的人(v1 接受此限制)。
        fallback = min(candidates, key=lambda d: _dist_from_frame_center(d['bbox']))

        # v2 的接縫:re-ID 上線後,連 fallback 都要過身分驗證;不過就回 None,
        # 由 _update_locked() 走「寧可原地等,也不要跟錯人」的路徑。
        if locked is not None:
            if self.identity_score(image, fallback['bbox'], locked) < cfg.identity_threshold:
                return None, False

        return fallback, False

    def _continuity_cost(self, bbox, prev_bbox):
        """與上一幀鎖定框的相似度成本,超出門檻回 None。"""
        cfg = self.config
        cx, cy = bbox_center(bbox)
        px, py = bbox_center(prev_bbox)
        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        if dist > cfg.track_center_max_dist:
            return None

        h = bbox_height(bbox)
        ph = bbox_height(prev_bbox)
        if h <= 0 or ph <= 0:
            return None
        ratio = h / ph if h > ph else ph / h
        if ratio > cfg.track_height_ratio_max:
            return None

        return dist + 0.5 * (ratio - 1.0)

    # -- 項目2:鎖定啟動條件 -----------------------------------------------
    def qualifies_for_lock(self, bbox):
        cfg = self.config
        if abs(center_offset(bbox)) > cfg.center_tolerance:
            return False
        if bbox_height(bbox) < cfg.min_bbox_height:
            return False
        return True

    # -- 內部:狀態處理 -----------------------------------------------------
    def _update_unlocked(self, det, image, now, events):
        """waiting / confirming。兩種狀態下機器人都原地不動,等目標自己站到中間。"""
        cfg = self.config

        # 鎖定那一關也要過身分驗證。
        #
        # 原本 identity_score 只在 select_target() 裡、而且只在「已經有鎖定目標」
        # 時被呼叫 —— 因為 re-ID 的本質是「跟已鎖定的人比對」,沒有比對對象就無從比起。
        # 但 v1 改用實體標記之後,標記是**絕對身分**,不需要參照對象,所以它也該
        # 守住鎖定這一關。少了這一行,任何人走到畫面中央站 0.75 秒都會被鎖定,
        # 身分驗證等於只在跟隨中生效,鎖定當下形同虛設。
        #
        # 第三個參數傳 None:預設實作本來就忽略它,MarkerVerifier 也不需要它。
        if (det is None
                or not self.qualifies_for_lock(det['bbox'])
                or self.identity_score(image, det['bbox'], None) < cfg.identity_threshold):
            if self.state == STATE_CONFIRMING:
                events.append(EVENT_CANCEL_CONFIRMING)
            self.state = STATE_WAITING
            self._confirm_started = None
            self._confirm_bbox = None
            return self._act(MOTOR_STOP, 0.0, None, events)

        if self.state != STATE_CONFIRMING:
            self.state = STATE_CONFIRMING
            self._confirm_started = now
            self._confirm_bbox = det['bbox']
            events.append(EVENT_ENTER_CONFIRMING)
            return self._act(MOTOR_STOP, 0.0, det, events)

        # 倒數期間也要確認是同一個人,不然換人站過來會接續倒數
        if self._continuity_cost(det['bbox'], self._confirm_bbox) is None:
            self._confirm_started = now
            events.append(EVENT_CANCEL_CONFIRMING)
            events.append(EVENT_ENTER_CONFIRMING)
        self._confirm_bbox = det['bbox']

        if now - self._confirm_started >= cfg.confirm_seconds:
            signature = self.signature_fn(image, det['bbox'])
            self._locked = LockedTarget(det['bbox'], signature, now)
            self.state = STATE_LOCKED
            self._confirm_started = None
            self._confirm_bbox = None
            events.append(EVENT_LOCK_ACQUIRED)
            return self._act(MOTOR_STEER, center_offset(det['bbox']), det, events)

        return self._act(MOTOR_STOP, 0.0, det, events)

    def _update_locked(self, det, matched_previous, now, events, distance_m=None):
        cfg = self.config

        if det is None:
            # 項目5:畫面中完全沒有候選框 → 不前進,不是官方的 robot.forward()
            return self._handle_lost(now, events)

        # 又看到目標了,取消搜尋
        self._search_started = None
        self._search_direction = 0

        if not matched_previous:
            # 項目4:重新鎖定。刻意不發任何蜂鳴器聲響(需求文件第 112 行)。
            events.append(EVENT_RELOCKED)

        self._locked.update(det['bbox'], now)
        return self._act(MOTOR_STEER, center_offset(det['bbox']), det, events,
                         speed_scale=self._speed_scale(det['bbox'], distance_m))

    # -- 跟隨距離控制 -------------------------------------------------------
    def _speed_scale(self, bbox, distance_m):
        """回傳 0.0~1.0 的前進速度倍率。0 代表停住但仍可轉向。"""
        cfg = self.config

        # 超音波優先,但只在目標大致位於正前方時才採信
        if (distance_m is not None
                and abs(center_offset(bbox)) <= cfg.distance_trust_offset):
            return _ramp(distance_m, cfg.stop_distance_m, cfg.follow_distance_m)

        # 退回框高。框越大代表越近,所以方向相反
        return _ramp(bbox_height(bbox), cfg.stop_bbox_height, cfg.slow_bbox_height)

    def expects_target_ahead(self):
        """正前方是否就是鎖定中的目標?

        給呼叫端判斷「超音波量到的近距離」該解讀成什麼用的:
        是的話代表人很近(該減速),不是的話才是障礙物(該避障)。
        少了這個判斷,機器人會把被跟隨的人當成障礙物繞開。
        """
        if self.state != STATE_LOCKED or self._locked is None:
            return False
        return abs(center_offset(self._locked.bbox)) <= self.config.distance_trust_offset

    def _handle_lost(self, now, events):
        """跟丟:朝最後已知的方向轉一小段,轉完仍找不到就原地等。"""
        cfg = self.config

        if now - self._locked.last_seen > cfg.lost_grace_seconds:
            self._locked = None
            self.state = STATE_WAITING
            self._search_started = None
            self._search_direction = 0
            events.append(EVENT_LOCK_LOST)
            return self._act(MOTOR_STOP, 0.0, None, events)

        # 跟丟的第一幀:用消失前的位置決定要不要轉、往哪轉
        if self._search_started is None:
            offset = center_offset(self._locked.bbox)
            self._search_started = now
            if abs(offset) >= cfg.search_min_offset:
                self._search_direction = 1 if offset > 0 else -1
                events.append(EVENT_SEARCHING)
            else:
                # 從正中央消失,方向資訊不可靠,寧可不動
                self._search_direction = 0

        if (self._search_direction != 0
                and now - self._search_started < cfg.search_turn_seconds):
            motor = MOTOR_TURN_RIGHT if self._search_direction > 0 else MOTOR_TURN_LEFT
            return self._act(motor, 0.0, None, events)

        return self._act(MOTOR_STOP, 0.0, None, events)

    def _handle_impassable(self, now, events):
        """項目7:無法跨越的障礙。暫停跟隨、通知照護者、保留鎖定等目標回來。"""
        cfg = self.config
        if self.state != STATE_IMPASSABLE:
            self.state = STATE_IMPASSABLE
            events.append(EVENT_ENTER_IMPASSABLE)
            # 清掉歷史,離開這個狀態後要重新累積一整個視窗才會再次判定為擺盪,
            # 否則剛恢復就會被舊資料立刻再判一次,在兩個狀態間彈跳。
            self._block_history.clear()
            self._detour_phase = None
            self._detour_advance_started = None
            self._detour_attempts = 0
            self._search_started = None
            self._search_direction = 0
        if self._locked is not None:
            self._locked.last_seen = now   # 暫停期間不消耗跟丟寬限
        if self._should_notify(self._last_impassable_notify, now, cfg.notify_cooldown):
            self._last_impassable_notify = now
            events.append(EVENT_NOTIFY_IMPASSABLE)
        return self._act(MOTOR_STOP, 0.0, None, events)

    def _track_absence(self, candidates, now, events):
        """項目5 的通知節流:連續找不到人超過 delay 秒才通知,且有 cooldown。"""
        cfg = self.config
        if candidates:
            self._absent_since = None
            return
        if self._absent_since is None:
            self._absent_since = now
            return
        if now - self._absent_since < cfg.no_target_notify_delay:
            return
        if self._should_notify(self._last_no_target_notify, now, cfg.notify_cooldown):
            self._last_no_target_notify = now
            events.append(EVENT_NOTIFY_NO_TARGET)

    @staticmethod
    def _should_notify(last, now, cooldown):
        return last is None or (now - last) >= cooldown

    def _act(self, motor, steer, det, events, speed_scale=1.0):
        return Action(state=self.state, motor=motor, steer=steer,
                      target=det, events=events, speed_scale=speed_scale)

    # -- 給 UI / debug 用 ---------------------------------------------------
    @property
    def locked_bbox(self):
        return self._locked.bbox if self._locked is not None else None

    def confirm_progress(self, now):
        """0.0~1.0,給 LED 閃爍或畫面提示用。"""
        if self.state != STATE_CONFIRMING or self._confirm_started is None:
            return 0.0
        return min(1.0, (now - self._confirm_started) / self.config.confirm_seconds)
