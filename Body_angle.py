import cv2, math, csv, numpy as np
import mediapipe as mp

VIDEO_PATH = "C:\\Users\\User\\Desktop\\IMG_6442.MOV"   # ← 改成你的影片
CSV_PATH   = "IMG_6442_angles.csv"

# ====== 角度判定範圍設定 ======
TOL = 2.0  # 單點目標的容差（度）
ranges = {
    "knee_flexion": (108, 112),
    "knee_extension": (35, 40),
    "back": (45 - TOL, 45 + TOL),                # 相對垂直
    "shoulder_to_elbow": None,                   # 未指定，僅計算
    "shoulder_to_wrist": (90 - 3, 90 + 3),       # 相對軀幹
    "elbow": (150, 170),
    "forearm": None                               # 未指定，僅計算
}

mp_pose = mp.solutions.pose
draw = mp.solutions.drawing_utils

def angle_abc(a, b, c):
    # 夾角 a-b-c（以 b 為頂點），回傳 0~180
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by], dtype=float)
    v2 = np.array([cx - bx, cy - by], dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return None
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosv))

def segment_angle(p, q, ref="vertical"):
    # 向量 p->q 相對垂直/水平的角度（0~180）
    vx, vy = q[0] - p[0], q[1] - p[1]
    if ref == "vertical":
        # 與 (0, -1) 的夾角
        dot = -vy; norm = math.hypot(vx, vy)
    else:  # "horizontal"
        dot = vx;  norm = math.hypot(vx, vy)
    if norm == 0: return None
    cosv = np.clip(dot / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosv))

def mid(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)

def pick_better_side(lm, w, h):
    # 以可見度挑選左右側中較可靠的一側
    L = mp_pose.PoseLandmark
    left_ids  = [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE]
    right_ids = [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE]
    lv = sum(lm[i.value].visibility for i in left_ids)
    rv = sum(lm[i.value].visibility for i in right_ids)
    return ("left", left_ids) if lv >= rv else ("right", right_ids)

def pt(lm, i, w, h):
    return (lm[i].x * w, lm[i].y * h)

cap = cv2.VideoCapture(VIDEO_PATH)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

rows = []
while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    out = {}
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        side, ids = pick_better_side(lm, w, h)
        L = mp_pose.PoseLandmark

        # 取關鍵點
        sh = pt(lm, (L.LEFT_SHOULDER if side=="left" else L.RIGHT_SHOULDER).value, w, h)
        el = pt(lm, (L.LEFT_ELBOW    if side=="left" else L.RIGHT_ELBOW   ).value, w, h)
        wr = pt(lm, (L.LEFT_WRIST    if side=="left" else L.RIGHT_WRIST   ).value, w, h)
        hp = pt(lm, (L.LEFT_HIP      if side=="left" else L.RIGHT_HIP     ).value, w, h)
        kn = pt(lm, (L.LEFT_KNEE     if side=="left" else L.RIGHT_KNEE    ).value, w, h)
        an = pt(lm, (L.LEFT_ANKLE    if side=="left" else L.RIGHT_ANKLE   ).value, w, h)
        sh_o = pt(lm, L.RIGHT_SHOULDER.value, w, h)
        sh_i = pt(lm, L.LEFT_SHOULDER.value,  w, h)
        hp_o = pt(lm, L.RIGHT_HIP.value,      w, h)
        hp_i = pt(lm, L.LEFT_HIP.value,       w, h)

        # 背部角度：骨盆中點→肩中點 相對垂直
        shoulder_mid = mid(sh_i, sh_o)
        hip_mid = mid(hp_i, hp_o)
        back_angle = segment_angle(hip_mid, shoulder_mid, ref="vertical")

        # 膝角（同一角度同時用來判定彎曲/伸展）
        knee_angle = angle_abc(hp, kn, an)

        # 肩角（到肘/到腕）
        shoulder_to_elbow = angle_abc(el, sh, hp)
        shoulder_to_wrist = angle_abc(wr, sh, hp)

        # 肘角
        elbow_angle = angle_abc(sh, el, wr)

        # 前臂相對水平
        forearm_angle = segment_angle(el, wr, ref="horizontal")

        # 判定函式
        def judge(name, value):
            rng = ranges[name]
            if value is None: return "NA"
            if rng is None:   return "—"
            low, high = rng
            return "OK" if (low <= value <= high) else "OUT"

        # 畫面標示
        overlay = [
            (f"Side: {side}", None),
            (f"Knee angle: {knee_angle:.1f}  [{judge('knee_flexion', knee_angle)} for flexion]  "
             f"[{judge('knee_extension', knee_angle)} for extension]", (30, 60)),
            (f"Back angle (vs vertical): {back_angle:.1f}  [{judge('back', back_angle)}]", (30, 90)),
            (f"Shoulder(elbow): {shoulder_to_elbow:.1f}  [{judge('shoulder_to_elbow', shoulder_to_elbow)}]", (30,120)),
            (f"Shoulder(wrist): {shoulder_to_wrist:.1f}  [{judge('shoulder_to_wrist', shoulder_to_wrist)}]", (30,150)),
            (f"Elbow: {elbow_angle:.1f}  [{judge('elbow', elbow_angle)}]", (30,180)),
            (f"Forearm vs horizontal: {forearm_angle:.1f}  [{judge('forearm', forearm_angle)}]", (30,210)),
        ]
        for text, pos in overlay[1:]:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if "OK" in text else (0,0,255), 2)

        draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out = dict(
            side=side,
            knee_angle=knee_angle,
            back_angle=back_angle,
            shoulder_to_elbow=shoulder_to_elbow,
            shoulder_to_wrist=shoulder_to_wrist,
            elbow_angle=elbow_angle,
            forearm_angle=forearm_angle
        )
        rows.append(out)

    cv2.imshow("Angles", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# 寫 CSV
if rows:
    keys = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {CSV_PATH}")
else:
    print("No pose detected; CSV not written.")
