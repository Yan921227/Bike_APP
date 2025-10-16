# 方案2：光流追蹤 + MediaPipe 混合方法
"""
核心思想：
1. 使用 MediaPipe 初始化關鍵點位置
2. 使用光流法追蹤關鍵點在連續幀間的移動
3. 定期用 MediaPipe 重新校正，避免漂移累積
4. 異常檢測：比較光流結果和 MediaPipe 結果，取可信度高的
"""

import cv2
import numpy as np
import mediapipe as mp

class OpticalFlowPoseTracker:
    def __init__(self):
        # MediaPipe 設定
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 光流法參數
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 狀態變數
        self.prev_frame = None
        self.prev_keypoints = None
        self.frame_count = 0
        self.recalibrate_interval = 30  # 每30幀重新用MediaPipe校正
        self.confidence_threshold = 0.7
        
        print("光流追蹤器初始化完成")
    
    def detect_initial_pose(self, frame):
        """使用 MediaPipe 檢測初始姿態"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                if landmark.visibility > self.confidence_threshold:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    keypoints.append([x, y])
                else:
                    keypoints.append(None)  # 低信心度的點設為None
            return np.array(keypoints, dtype=object)
        return None
    
    def track_with_optical_flow(self, current_frame):
        """使用光流法追蹤關鍵點"""
        if self.prev_frame is None or self.prev_keypoints is None:
            return self.detect_initial_pose(current_frame)
        
        # 準備光流追蹤的點
        valid_prev_points = []
        valid_indices = []
        
        for i, kp in enumerate(self.prev_keypoints):
            if kp is not None:
                valid_prev_points.append([kp[0], kp[1]])
                valid_indices.append(i)
        
        if len(valid_prev_points) == 0:
            return self.detect_initial_pose(current_frame)
        
        valid_prev_points = np.array(valid_prev_points, dtype=np.float32)
        
        # 計算光流
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, current_gray, valid_prev_points, None, **self.lk_params
        )
        
        # 更新關鍵點位置
        updated_keypoints = self.prev_keypoints.copy()
        
        for i, (idx, new_point, st, err) in enumerate(zip(valid_indices, new_points, status, error)):
            if st == 1 and err < 30:  # 追蹤成功且誤差小
                updated_keypoints[idx] = new_point
            # 追蹤失敗的點保持原位置或設為None
        
        return updated_keypoints
    
    def hybrid_detection(self, frame):
        """混合檢測：光流 + 定期 MediaPipe 校正"""
        self.frame_count += 1
        
        # 定期重新用 MediaPipe 檢測以避免累積誤差
        if self.frame_count % self.recalibrate_interval == 0 or self.prev_keypoints is None:
            print(f"第 {self.frame_count} 幀：重新校正")
            keypoints = self.detect_initial_pose(frame)
        else:
            # 使用光流追蹤
            keypoints = self.track_with_optical_flow(frame)
            
            # 可選：與 MediaPipe 結果做比較驗證
            if self.frame_count % 10 == 0:  # 每10幀驗證一次
                mp_keypoints = self.detect_initial_pose(frame)
                keypoints = self.validate_with_mediapipe(keypoints, mp_keypoints)
        
        # 更新狀態
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        
        return keypoints
    
    def validate_with_mediapipe(self, flow_keypoints, mp_keypoints):
        """用 MediaPipe 結果驗證光流結果"""
        if mp_keypoints is None:
            return flow_keypoints
        
        validated_keypoints = flow_keypoints.copy()
        
        for i in range(len(flow_keypoints)):
            if flow_keypoints[i] is not None and mp_keypoints[i] is not None:
                # 計算兩個結果之間的距離
                flow_point = flow_keypoints[i]
                mp_point = mp_keypoints[i]
                distance = np.linalg.norm(flow_point - mp_point)
                
                # 如果距離太大，使用 MediaPipe 結果
                if distance > 50:  # 50像素閾值
                    validated_keypoints[i] = mp_point
                    
        return validated_keypoints
    
    def draw_keypoints(self, frame, keypoints):
        """繪製關鍵點"""
        if keypoints is None:
            return frame
        
        for i, kp in enumerate(keypoints):
            if kp is not None:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame

def test_optical_flow_tracking():
    """測試光流追蹤方法"""
    tracker = OpticalFlowPoseTracker()
    
    # 使用你的影片
    cap = cv2.VideoCapture("C:\\Users\\User\\Desktop\\IMG_1891.MOV")
    
    # 設定輸出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('optical_flow_tracking.mp4', fourcc, 30.0, (1080, 720))
    
    print("開始光流追蹤測試...")
    print("按 q 鍵退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1080, 720))
        
        # 使用混合檢測
        keypoints = tracker.hybrid_detection(frame)
        
        # 繪製結果
        result_frame = tracker.draw_keypoints(frame, keypoints)
        
        # 顯示和保存
        cv2.imshow('Optical Flow + MediaPipe', result_frame)
        out.write(result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("光流追蹤測試完成！輸出：optical_flow_tracking.mp4")

if __name__ == "__main__":
    print("=== 光流追蹤 + MediaPipe 混合方法 ===")
    print("這個方法結合了光流的連續性和 MediaPipe 的準確性")
    print()
    test_optical_flow_tracking()