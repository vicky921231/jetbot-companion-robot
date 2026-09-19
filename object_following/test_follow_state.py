"""狀態機的離線測試。在筆電上直接跑,不需要 JetBot、不需要 CUDA:

    python test_follow_state.py

時間是餵進去的參數(不是 time.time()),所以「連續 0.75 秒」這種條件
可以在毫秒內測完,不用真的等。
"""

import follow_state as fs


def det(cx, cy=0.5, h=0.4, w=0.2, label=1):
    """用中心點與高度組出一個偵測框,省得每次手算四個角。"""
    return {'label': label,
            'bbox': [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]}


def make_sm(**overrides):
    cfg = fs.FollowConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return fs.FollowStateMachine(cfg)


def check(name, cond):
    print(('  PASS  ' if cond else '  FAIL  ') + name)
    if not cond:
        raise AssertionError(name)


# --- 項目2:鎖定啟動 ------------------------------------------------------

def test_lock_requires_centered_and_close():
    print('項目2 鎖定啟動:必須置中 + 夠近 + 連續 0.75 秒')
    sm = make_sm()

    # 人在畫面左邊(偏離 > 15%)→ 不進入 confirming
    a = sm.update([det(0.25)], None, 0.0)
    check('偏離中心不進入 confirming', a.state == fs.STATE_WAITING)
    check('未鎖定時原地不動', a.motor == fs.MOTOR_STOP)

    # 置中但太遠(框高不足)→ 仍不進入 confirming
    a = sm.update([det(0.5, h=0.1)], None, 0.1)
    check('框高不足不進入 confirming', a.state == fs.STATE_WAITING)

    # 置中 + 夠近 → confirming,但還沒滿 0.75 秒
    a = sm.update([det(0.5)], None, 1.0)
    check('進入 confirming', a.state == fs.STATE_CONFIRMING)
    check('confirming 期間原地不動', a.motor == fs.MOTOR_STOP)
    a = sm.update([det(0.5)], None, 1.5)
    check('0.5 秒還不夠', a.state == fs.STATE_CONFIRMING)

    # 滿 0.75 秒 → locked,且這是唯一該響蜂鳴器的時刻
    a = sm.update([det(0.5)], None, 1.8)
    check('滿 0.75 秒鎖定', a.state == fs.STATE_LOCKED)
    check('鎖定發出 lock_acquired', fs.EVENT_LOCK_ACQUIRED in a.events)
    check('鎖定後開始差速跟隨', a.motor == fs.MOTOR_STEER)


def test_confirming_resets_when_target_leaves():
    print('項目2 鎖定啟動:倒數中目標離開要重新計時')
    sm = make_sm()
    sm.update([det(0.5)], None, 0.0)
    a = sm.update([], None, 0.3)
    check('目標消失取消倒數', a.state == fs.STATE_WAITING)
    check('發出 cancel_confirming', fs.EVENT_CANCEL_CONFIRMING in a.events)
    sm.update([det(0.5)], None, 0.4)
    a = sm.update([det(0.5)], None, 0.9)
    check('重新倒數而不是接續舊計時', a.state == fs.STATE_CONFIRMING)


# --- 項目1:跨幀連續性追蹤 ------------------------------------------------

def test_tracking_prefers_continuity_over_center():
    print('項目1 連續性追蹤:延續上一幀鎖定框,不是每幀重選最近的')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)
    check('先鎖定成功', sm.state == fs.STATE_LOCKED)

    # 目標逐幀往右漂(實際 fps 下人不會瞬移),鎖定框跟著移動
    for cx, t in ((0.54, 1.0), (0.58, 1.1), (0.62, 1.2)):
        sm.update([det(cx)], None, t)

    # 此時有個路人站到畫面正中央 0.50,目標在 0.64。
    # 官方邏輯會選正中央那個(離中心最近);我們要選延續的那個。
    a = sm.update([det(0.64), det(0.50, h=0.42)], None, 1.3)
    target_cx = fs.bbox_center(a.target['bbox'])[0]
    check('選到延續框而非離中心最近的框', abs(target_cx - 0.64) < 1e-6)
    check('延續成功不算 relock', fs.EVENT_RELOCKED not in a.events)


# --- 項目4:跟丟後退而選最近候選框 -----------------------------------------

def test_relock_falls_back_to_nearest():
    print('項目4 跟丟:退而選最近候選框,且不觸發蜂鳴器事件')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    # 原目標消失,畫面上只剩一個位置差很遠的人 → 幾何比對對不上
    a = sm.update([det(0.9, h=0.42)], None, 1.0)
    check('仍維持 locked(不中斷跟隨)', a.state == fs.STATE_LOCKED)
    check('發出 relocked', fs.EVENT_RELOCKED in a.events)
    check('relock 不得發出 lock_acquired(不響蜂鳴器)',
          fs.EVENT_LOCK_ACQUIRED not in a.events)


def test_reid_seam_rejects_stranger():
    print('接縫驗證:換上 identity_score 後,跟丟就不會亂認人')
    sm = make_sm(identity_threshold=0.5)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    sm.identity_score = lambda image, bbox, locked: 0.0   # 模擬 v2 判定「不是同一人」
    a = sm.update([det(0.9, h=0.42)], None, 1.0)
    check('身分不符時不重新鎖定', a.target is None)
    check('身分不符時原地等待', a.motor == fs.MOTOR_STOP)


# --- 身分標記:鎖定關卡 ---------------------------------------------------
#
# v1 用螢光雙色貼紙取代 re-ID(見 marker.py)。標記是絕對身分,不需要參照對象,
# 所以它必須也守住「鎖定」這一關,不能只在跟隨中生效。

def test_lock_requires_identity():
    print('身分標記:沒貼標記的人即使站到正中央也不會被鎖定')
    sm = make_sm(identity_threshold=0.5)
    sm.identity_score = lambda image, bbox, locked: 0.0   # 身上沒有標記

    for t in (0.0, 0.4, 0.8, 1.2):
        a = sm.update([det(0.5)], None, t)
    check('全程停在 waiting', a.state == fs.STATE_WAITING)
    check('沒有發出 lock_acquired', fs.EVENT_LOCK_ACQUIRED not in a.events)
    check('原地不動', a.motor == fs.MOTOR_STOP)


def test_lock_proceeds_with_identity():
    print('身分標記:貼了標記就照原本的三個條件鎖定')
    sm = make_sm(identity_threshold=0.5)
    sm.identity_score = lambda image, bbox, locked: 1.0   # 看得到標記

    a = sm.update([det(0.5)], None, 0.0)
    check('進入 confirming', a.state == fs.STATE_CONFIRMING)
    a = sm.update([det(0.5)], None, 0.8)
    check('滿 0.75 秒仍然鎖定得了', a.state == fs.STATE_LOCKED)
    check('鎖定發出 lock_acquired', fs.EVENT_LOCK_ACQUIRED in a.events)


def test_identity_checked_before_lock_not_after_grace():
    print('身分標記:鎖定關卡拿到的 locked_target 是 None(寬限不該適用於此)')
    seen = []
    sm = make_sm(identity_threshold=0.5)

    def spy(image, bbox, locked):
        seen.append(locked)
        return 1.0

    sm.identity_score = spy
    sm.update([det(0.5)], None, 0.0)
    check('未鎖定時第三個參數為 None', seen and seen[0] is None)

    sm.update([det(0.5)], None, 0.8)     # 鎖定
    seen[:] = []
    sm.update([det(0.5)], None, 0.9)     # 已鎖定,走 select_target 那條路
    check('已鎖定後第三個參數是 LockedTarget',
          seen and isinstance(seen[0], fs.LockedTarget))


# --- 項目5:畫面中完全沒人 ------------------------------------------------

def test_no_candidates_stops_and_notifies():
    print('項目5 沒人在鏡頭前:原地等待(不前進)+ 通知照護者')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    a = sm.update([], None, 1.0)
    check('沒候選框時停住,不是官方的 forward()', a.motor == fs.MOTOR_STOP)
    check('寬限期內保留鎖定', a.state == fs.STATE_LOCKED)

    a = sm.update([], None, 4.5)   # 超過 no_target_notify_delay
    check('連續找不到人才通知', fs.EVENT_NOTIFY_NO_TARGET in a.events)
    check('超過寬限期放棄鎖定', a.state == fs.STATE_WAITING)

    a = sm.update([], None, 5.0)
    check('通知有 cooldown,不會每幀狂送', fs.EVENT_NOTIFY_NO_TARGET not in a.events)


# --- 項目6:避障結束銜接回原鎖定 -------------------------------------------

def test_distance_keeping_slows_and_stops():
    print('跟隨距離:靠超音波減速,太近就停,不再一路逼近')
    sm = make_sm(follow_distance_m=0.80, stop_distance_m=0.45)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t, distance_m=1.5)

    a = sm.update([det(0.5)], None, 1.0, distance_m=1.5)
    check('距離夠遠時全速', a.speed_scale == 1.0)

    a = sm.update([det(0.5)], None, 1.1, distance_m=0.625)   # 正好在中間
    check('進入減速區時速度減半', abs(a.speed_scale - 0.5) < 0.02)

    a = sm.update([det(0.5)], None, 1.2, distance_m=0.40)
    check('太近時速度歸零', a.speed_scale == 0.0)
    check('但仍維持跟隨狀態', a.motor == fs.MOTOR_STEER)
    check('仍保有轉向資訊,鏡頭可以繼續對著人',
          a.target is not None)


def test_distance_falls_back_to_bbox_height():
    print('跟隨距離:沒有超音波時退回用框高當距離代理')
    sm = make_sm(slow_bbox_height=0.55, stop_bbox_height=0.75)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5, h=0.4)], None, t)

    a = sm.update([det(0.5, h=0.4)], None, 1.0)          # 框小 = 遠
    check('框小時全速', a.speed_scale == 1.0)

    a = sm.update([det(0.5, h=0.65)], None, 1.1)         # 框變大 = 變近
    check('框變大時減速', abs(a.speed_scale - 0.5) < 0.02)

    a = sm.update([det(0.5, h=0.8)], None, 1.2)          # 框很大 = 很近
    check('框很大時停住', a.speed_scale == 0.0)


def test_sonar_ignored_when_target_off_centre():
    print('跟隨距離:目標偏到側邊時不採信超音波(波束打到的不是他)')
    sm = make_sm(stop_distance_m=0.45, distance_trust_offset=0.15,
                 slow_bbox_height=0.55, stop_bbox_height=0.75)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)
    for cx, t in ((0.6, 1.0), (0.72, 1.1)):
        sm.update([det(cx)], None, t)

    # 目標偏到 0.72(偏移 0.22 > 0.15),超音波卻讀到 0.2 公尺——那是牆不是人
    a = sm.update([det(0.72, h=0.4)], None, 1.2, distance_m=0.2)
    check('不因為側邊的牆而停住', a.speed_scale == 1.0)


def test_expects_target_ahead():
    print('跟隨距離:正前方是不是鎖定目標——決定近距離該解讀成人還是障礙物')
    sm = make_sm(distance_trust_offset=0.15)
    check('還沒鎖定時為否', sm.expects_target_ahead() is False)

    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)
    check('鎖定且置中時為是', sm.expects_target_ahead() is True)

    for cx, t in ((0.6, 1.0), (0.72, 1.1)):
        sm.update([det(cx)], None, t)
    check('目標偏到側邊時為否', sm.expects_target_ahead() is False)


def test_search_turns_toward_last_known_direction():
    print('跟丟搜尋:直角牆角——目標往右消失就往右轉,不是立刻停住')
    sm = make_sm(search_turn_seconds=1.5, search_min_offset=0.12)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    # 目標往右漂,像是準備轉彎
    for cx, t in ((0.58, 1.0), (0.66, 1.1), (0.74, 1.2)):
        sm.update([det(cx)], None, t)

    # 轉過牆角,被牆擋住而突然消失
    a = sm.update([], None, 1.3)
    check('跟丟後往右轉,不是停住', a.motor == fs.MOTOR_TURN_RIGHT)
    check('發出 searching', fs.EVENT_SEARCHING in a.events)
    check('鎖定仍保留', a.state == fs.STATE_LOCKED)
    check('搜尋不觸發蜂鳴器', fs.EVENT_LOCK_ACQUIRED not in a.events)

    a = sm.update([], None, 2.0)
    check('搜尋期間持續轉', a.motor == fs.MOTOR_TURN_RIGHT)


def test_search_goes_left_when_target_left():
    print('跟丟搜尋:方向要跟著目標,不是固定往左')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)
    for cx, t in ((0.42, 1.0), (0.34, 1.1), (0.26, 1.2)):
        sm.update([det(cx)], None, t)

    a = sm.update([], None, 1.3)
    check('目標往左消失就往左轉', a.motor == fs.MOTOR_TURN_LEFT)


def test_search_stays_still_when_target_vanishes_from_centre():
    print('跟丟搜尋:從正中央消失時不亂轉')
    sm = make_sm(search_min_offset=0.12)
    for t in (0.0, 0.4, 0.8, 1.0):
        sm.update([det(0.5)], None, t)   # 一直在正中央

    a = sm.update([], None, 1.1)
    check('沒有方向線索就原地不動', a.motor == fs.MOTOR_STOP)
    check('不發出 searching', fs.EVENT_SEARCHING not in a.events)


def test_search_reacquires_and_resumes():
    print('跟丟搜尋:轉到看見目標就立刻接回跟隨')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)
    for cx, t in ((0.6, 1.0), (0.7, 1.1)):
        sm.update([det(cx)], None, t)

    a = sm.update([], None, 1.2)
    check('先開始轉', a.motor == fs.MOTOR_TURN_RIGHT)

    a = sm.update([det(0.72)], None, 1.5)   # 轉到又看見了
    check('看見就恢復差速跟隨', a.motor == fs.MOTOR_STEER)
    check('不需要重新站定', a.state == fs.STATE_LOCKED)


def test_search_does_not_extend_grace():
    print('跟丟搜尋:搜尋不會延長寬限,該放棄還是要放棄')
    sm = make_sm(lost_grace_seconds=2.0, search_turn_seconds=1.5)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)
    sm.update([det(0.7)], None, 1.0)

    a = sm.update([], None, 2.0)
    check('寬限內仍在搜尋或等待', a.state == fs.STATE_LOCKED)

    a = sm.update([], None, 3.5)   # 超過 1.0 + 2.0 的寬限
    check('超過寬限就解除鎖定', a.state == fs.STATE_WAITING)
    check('發出 lock_lost', fs.EVENT_LOCK_LOST in a.events)
    check('放棄後停住', a.motor == fs.MOTOR_STOP)


def test_blocked_preserves_lock():
    print('項目6 避障:結束後銜接回上一次鎖定框,不重跑鎖定流程')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    for t in (1.0, 2.0, 3.0):   # 避障 3 秒,超過 2 秒跟丟寬限
        a = sm.on_blocked(t)
        check('避障期間左轉', a.motor == fs.MOTOR_TURN_LEFT)

    a = sm.update([det(0.55)], None, 3.1)   # 轉開了,進入橫移階段
    check('轉開後進入橫移', a.motor == fs.MOTOR_FORWARD)

    a = sm.update([det(0.55)], None, 5.0)   # 橫移 1.5 秒走完
    check('繞行後仍是 locked', a.state == fs.STATE_LOCKED)
    check('繞行後直接跟隨,不用重新站定', a.motor == fs.MOTOR_STEER)
    check('繞行後不算 relock', fs.EVENT_RELOCKED not in a.events)


def test_persistent_block_escalates_to_impassable():
    print('項目7 無法跨越:持續被擋超過門檻就升級,不需要額外的樓梯感測器')
    sm = make_sm(impassable_after_blocked_seconds=5.0)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    a = sm.on_blocked(1.0)
    check('剛被擋住時照官方左轉繞開', a.motor == fs.MOTOR_TURN_LEFT)
    a = sm.on_blocked(4.0)
    check('還沒到門檻時繼續繞', a.motor == fs.MOTOR_TURN_LEFT)

    a = sm.on_blocked(6.5)   # 已連續被擋 5.5 秒
    check('持續被擋超過門檻升級成 impassable_wait', a.state == fs.STATE_IMPASSABLE)
    check('升級後停止,不再原地空轉', a.motor == fs.MOTOR_STOP)
    check('通知照護者', fs.EVENT_NOTIFY_IMPASSABLE in a.events)
    check('不對目標者發警示音', fs.EVENT_LOCK_ACQUIRED not in a.events)

    a = sm.update([det(0.5)], None, 7.0)
    check('障礙排除後恢復跟隨', fs.EVENT_RESUME_FOLLOW in a.events)
    check('且仍鎖在原目標身上', a.state == fs.STATE_LOCKED)

    a = sm.on_blocked(7.5)
    check('重新被擋時計時歸零,先繞再說', a.motor == fs.MOTOR_TURN_LEFT)


def test_oscillation_is_detected_as_impassable():
    print('項目7 擺盪:左轉是原地自轉,障礙擋在中間會來回擺盪,也要判定為繞不過去')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    # 模擬石頭擋在機器人與目標之間:
    # 左轉到不擋住 → 跟隨邏輯把車頭拉回目標 → 又正對石頭 → 再左轉……
    # 每次被擋都只有 0.5 秒,遠不到「連續 5 秒」的門檻。
    escalated_at = None
    for i in range(24):
        t = 1.0 + i * 0.5
        if i % 2 == 0:
            a = sm.on_blocked(t)              # 被石頭擋住
        else:
            a = sm.update([det(0.5)], None, t)  # 轉開了,看得到目標
        if a.state == fs.STATE_IMPASSABLE:
            escalated_at = t
            break

    check('擺盪會被判定為繞不過去,不會無限卡住', escalated_at is not None)
    # 每次被擋只有 0.5 秒,「連續被擋 5 秒」的門檻永遠不可能達成,
    # 所以能判定出來一定是靠滑動視窗的擺盪偵測。
    check('在一個視窗內判定完成(不會拖到天荒地老)',
          escalated_at is not None and escalated_at - 1.0 <= sm.config.block_window_seconds)
    print('      (擺盪 %.1f 秒後判定)' % (escalated_at - 1.0))


def test_normal_corner_turn_is_not_mistaken_for_impassable():
    print('項目7 誤判防護:單純轉個彎繞開,不該被當成死路')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    for t in (1.0, 1.5, 2.0, 2.5, 3.0):   # 被擋 2 秒後成功繞開
        a = sm.on_blocked(t)
        check('繞行期間維持左轉', a.motor == fs.MOTOR_TURN_LEFT)

    states = []
    for i in range(20):                    # 接著順利跟隨 10 秒
        a = sm.update([det(0.5)], None, 3.5 + i * 0.5)
        check('繞開後不誤判為死路', a.state != fs.STATE_IMPASSABLE)
        states.append(a.state)
    check('橫移結束後回到正常跟隨', states[-1] == fs.STATE_LOCKED)


def test_detour_commits_before_rejoining():
    print('繞行:轉開後強制橫移一段才交還跟隨,這才是修掉擺盪的關鍵')
    sm = make_sm(detour_advance_seconds=1.5)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    a = sm.on_blocked(1.0)
    check('被擋住時進入繞行', a.state == fs.STATE_DETOUR)
    check('繞行第一階段是原地左轉', a.motor == fs.MOTOR_TURN_LEFT)
    check('發出 enter_detour', fs.EVENT_ENTER_DETOUR in a.events)

    # 轉開了。這裡是關鍵:目標就在正前方 0.5,官方邏輯會立刻轉回去正對障礙物,
    # 我們則必須先把橫移走完。
    a = sm.update([det(0.5)], None, 1.5)
    check('轉開後不立刻轉回目標', a.motor != fs.MOTOR_STEER)
    check('改為直行橫移', a.motor == fs.MOTOR_FORWARD)
    check('發出 detour_advance', fs.EVENT_DETOUR_ADVANCE in a.events)

    a = sm.update([det(0.5)], None, 2.5)
    check('橫移未滿時繼續直行', a.motor == fs.MOTOR_FORWARD)

    a = sm.update([det(0.5)], None, 3.1)   # 橫移滿 1.5 秒
    check('橫移完成後交還跟隨邏輯', a.motor == fs.MOTOR_STEER)
    check('發出 detour_done', fs.EVENT_DETOUR_DONE in a.events)
    check('鎖定全程保留', a.state == fs.STATE_LOCKED)


def test_detour_advance_freezes_lost_timer():
    print('繞行:橫移期間目標暫時出框不該被判定跟丟')
    sm = make_sm(detour_advance_seconds=1.5, lost_grace_seconds=2.0)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    sm.on_blocked(1.0)
    for t in (1.5, 2.0, 2.5):      # 橫移期間目標完全不在畫面中
        a = sm.update([], None, t)
        check('橫移期間維持直行', a.motor == fs.MOTOR_FORWARD)

    a = sm.update([det(0.5)], None, 3.1)
    check('目標回到畫面後直接續跟,沒被判定跟丟', a.state == fs.STATE_LOCKED)
    check('不需要重新站定鎖定', a.motor == fs.MOTOR_STEER)


def test_repeated_detours_give_up():
    print('繞行:同一個障礙繞不過去就放棄,不無限嘗試')
    sm = make_sm(detour_advance_seconds=1.0, max_detour_attempts=3,
                 detour_reset_seconds=10.0)
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    # 繞行 → 橫移完成 → 又被擋住,重複四輪
    t = 1.0
    gave_up = False
    for attempt in range(4):
        a = sm.on_blocked(t)
        if a.state == fs.STATE_IMPASSABLE:
            gave_up = True
            break
        t += 0.5
        sm.update([det(0.5)], None, t)      # 轉開,開始橫移
        t += 1.2
        sm.update([det(0.5)], None, t)      # 橫移完成
        t += 0.5

    check('繞行次數用盡後判定繞不過去', gave_up)
    check('放棄時停下來', sm.state == fs.STATE_IMPASSABLE)


# --- 項目7:無法跨越的障礙 ------------------------------------------------

def test_impassable_pauses_without_buzzer():
    print('項目7 無法跨越障礙:暫停 + 通知照護者,不對目標者發警示音')
    sm = make_sm()
    for t in (0.0, 0.4, 0.8):
        sm.update([det(0.5)], None, t)

    a = sm.update([det(0.5)], None, 1.0, impassable=True)
    check('進入 impassable_wait', a.state == fs.STATE_IMPASSABLE)
    check('暫停跟隨', a.motor == fs.MOTOR_STOP)
    check('通知照護者', fs.EVENT_NOTIFY_IMPASSABLE in a.events)
    check('不對目標者發警示音', fs.EVENT_LOCK_ACQUIRED not in a.events)

    a = sm.update([det(0.5)], None, 1.5)
    check('障礙解除恢復跟隨', fs.EVENT_RESUME_FOLLOW in a.events)
    check('恢復後直接續跟原目標', a.state == fs.STATE_LOCKED and a.motor == fs.MOTOR_STEER)


# --- 全域規則:蜂鳴器只在鎖定完成響 ----------------------------------------

def test_buzzer_only_on_lock():
    print('全域規則:整段流程中 lock_acquired 只出現在真正鎖定的那一幀')
    sm = make_sm()
    timeline = [
        ([], 0.0), ([det(0.25)], 0.2), ([det(0.5)], 0.4), ([det(0.5)], 0.8),
        ([det(0.5)], 1.2), ([det(0.9, h=0.42)], 1.4), ([], 1.6), ([], 4.0),
    ]
    beeps = 0
    for candidates, t in timeline:
        a = sm.update(candidates, None, t)
        if fs.EVENT_LOCK_ACQUIRED in a.events:
            beeps += 1
    check('整段流程只響一次', beeps == 1)


if __name__ == '__main__':
    tests = [
        test_lock_requires_centered_and_close,
        test_confirming_resets_when_target_leaves,
        test_tracking_prefers_continuity_over_center,
        test_relock_falls_back_to_nearest,
        test_reid_seam_rejects_stranger,
        test_lock_requires_identity,
        test_lock_proceeds_with_identity,
        test_identity_checked_before_lock_not_after_grace,
        test_no_candidates_stops_and_notifies,
        test_distance_keeping_slows_and_stops,
        test_distance_falls_back_to_bbox_height,
        test_sonar_ignored_when_target_off_centre,
        test_expects_target_ahead,
        test_search_turns_toward_last_known_direction,
        test_search_goes_left_when_target_left,
        test_search_stays_still_when_target_vanishes_from_centre,
        test_search_reacquires_and_resumes,
        test_search_does_not_extend_grace,
        test_blocked_preserves_lock,
        test_persistent_block_escalates_to_impassable,
        test_oscillation_is_detected_as_impassable,
        test_normal_corner_turn_is_not_mistaken_for_impassable,
        test_detour_commits_before_rejoining,
        test_detour_advance_freezes_lost_timer,
        test_repeated_detours_give_up,
        test_impassable_pauses_without_buzzer,
        test_buzzer_only_on_lock,
    ]
    for t in tests:
        t()
    print('\n全部通過(%d 組)' % len(tests))
