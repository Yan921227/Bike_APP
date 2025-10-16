# 整合光流追蹤的姿態分析系統
import cv2
import mediapipe as mp
import math
import time
import statistics as sta
import mysql.connector
import numpy as np

class OpticalFlowBikeFit():
    mp_pose = mp.solutions.pose  # mediapipe 姿勢偵測

    rightPoint = [16, 14, 12, 24, 26, 28, 32, 30]
    leftPoint = [15, 13, 11, 23, 25, 27, 31, 29]

    rightAnglePoint = [12, 14, 24, 26, 28]
    leftAnglePoint = [11, 13, 23, 25, 27]

    datas = {11: [], 12: [], 13: [], 14: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: []}
    time_interval = 10  # 設置每10秒的時間間隔
    start_time = None
    
    # 數據庫連接變數
    conn = None
    cursor = None

    angleMap = {11: "肩", 12: "肩",
                13: "手肘", 14: "手肘",
                15: "手腕", 16: "手腕",
                23: "髖", 24: "髖",
                25: "膝", 26: "膝",
                27: "腳踝", 28: "腳踝"}

    enAngleMap = {11: "shoulder", 12: "shoulder",
                  13: "elbow", 14: "elbow",
                  15: "wrist", 16: "wrist",
                  23: "hip", 24: "hip",
                  25: "knee", 26: "knee",
                  27: "ankle", 28: "ankle"}

    def __init__(self):
        self.start_time = time.time()  # 初始化開始時間
        
        # 光流追蹤相關參數
        self.setup_optical_flow()
        
        # 初始化數據庫連接（可選）
        self.conn = None
        self.cursor = None
        try:
            # 如果需要資料庫功能，請修改以下連接資訊
            # self.conn = mysql.connector.connect(
            #     host='localhost',
            #     user='your_username',
            #     password='your_password',
            #     database='your_database'
            # )
            # self.cursor = self.conn.cursor()
            # print("數據庫連接成功")
            print("數據庫功能已停用，僅顯示角度統計")
        except Exception as err:
            print(f"數據庫連接失敗: {err}")
            self.conn = None
            self.cursor = None

    def setup_optical_flow(self):
        """設定光流追蹤參數"""
        # MediaPipe 設定
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
        self.confidence_threshold = 0.5
        
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
        
        return updated_keypoints
    
    def hybrid_detection(self, frame):
        """混合檢測：光流 + 定期 MediaPipe 校正"""
        self.frame_count += 1
        
        # 定期重新用 MediaPipe 檢測以避免累積誤差
        if self.frame_count % self.recalibrate_interval == 0 or self.prev_keypoints is None:
            keypoints = self.detect_initial_pose(frame)
        else:
            # 使用光流追蹤
            keypoints = self.track_with_optical_flow(frame)
        
        # 更新狀態
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        
        return keypoints

    # 找前後元素
    def find_neighboring_elements(self, number, elements_list):
        index = elements_list.index(number)
        previous_element = elements_list[index - 1] if index > 0 else None
        next_element = elements_list[index + 1] if index < len(elements_list) - 1 else None
        return previous_element, next_element

    # 計算角度的公式
    def calculate_angle(self, point1, point2, point3):
        if point1 is None or point2 is None or point3 is None:
            return 0.0
            
        # 將 numpy array 轉換為座標
        if isinstance(point1, np.ndarray):
            point1 = (point1[0], point1[1])
        if isinstance(point2, np.ndarray):
            point2 = (point2[0], point2[1])
        if isinstance(point3, np.ndarray):
            point3 = (point3[0], point3[1])
            
        vector_a = (point1[0] - point2[0], point1[1] - point2[1])
        vector_b = (point3[0] - point2[0], point3[1] - point2[1])
        dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        magnitude_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)
        magnitude_b = math.sqrt(vector_b[0] ** 2 + vector_b[1] ** 2)
        
        # 數值穩健性：避免除以零或超出 -1~1 範圍導致 math domain error
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        cos_val = dot_product / (magnitude_a * magnitude_b)
        cos_val = max(-1.0, min(1.0, cos_val))
        angle_radians = math.acos(cos_val)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    # 人體姿態檢測 - 光流追蹤版本
    def mpPose(self, cap, side='right', output_path=None):
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        # 如果要輸出影片，建立 VideoWriter
        writer = None
        if output_path is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (1080, 720))

        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            img = cv2.resize(img, (1080, 720))
            
            # 使用光流追蹤獲取關鍵點
            keypoints = self.hybrid_detection(img)

            if keypoints is not None:
                # 轉換 keypoints 為整數座標列表
                coord_list = []
                for kp in keypoints:
                    if kp is not None and isinstance(kp, (list, np.ndarray)) and len(kp) >= 2:
                        coord_list.append((int(kp[0]), int(kp[1])))
                    else:
                        coord_list.append((0, 0))
                
                # 繪製所有關鍵點（像 optical_flow_tracker.py 那樣）
                for i, kp in enumerate(keypoints):
                    if kp is not None and isinstance(kp, (list, np.ndarray)):
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                        cv2.putText(img, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # 計算角度 (使用與 main2dPose.py 相同的邏輯)
                if side == 'right':
                    for i in self.rightAnglePoint:
                        test1, test2 = self.find_neighboring_elements(i, self.rightPoint)
                        if (test1 is not None and test2 is not None and
                            i < len(coord_list) and test1 < len(coord_list) and test2 < len(coord_list) and
                            coord_list[i] != (0, 0) and coord_list[test1] != (0, 0) and coord_list[test2] != (0, 0)):
                            angle = self.calculate_angle(coord_list[test1], coord_list[i], coord_list[test2])
                            if angle > 0:  # 只記錄有效角度
                                self.datas[i].append(angle)
                else:
                    for i in self.leftAnglePoint:
                        test1, test2 = self.find_neighboring_elements(i, self.leftPoint)
                        if (test1 is not None and test2 is not None and
                            i < len(coord_list) and test1 < len(coord_list) and test2 < len(coord_list) and
                            coord_list[i] != (0, 0) and coord_list[test1] != (0, 0) and coord_list[test2] != (0, 0)):
                            angle = self.calculate_angle(coord_list[test1], coord_list[i], coord_list[test2])
                            if angle > 0:  # 只記錄有效角度
                                self.datas[i].append(angle)

            # 每 10 秒統計一次角度
            if time.time() - self.start_time >= self.time_interval:
                print("\n每 10 秒統計角度：")
                self.record_angle_statistics(side)
                self.start_time = time.time()  # 重置開始時間
                self.clear_data()

            # 顯示並寫入影片（若有指定）
            cv2.imshow('Optical Flow Pose Analysis', img)
            if writer is not None:
                writer.write(img)

            # 檢查按鍵，按 q 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("按下 q 鍵，正在退出...")
                break
        
        # 迴圈結束，釋放資源
        if writer is not None:
            writer.release()
            print(f"影片已儲存")
        cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗

    # 統計並顯示角度數據，並存入資料庫
    def record_angle_statistics(self, side):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

        if side == 'right':
            print("\n右側角度統計：")   
            for i in self.rightAnglePoint:
                if self.datas[i]:  # 確認列表不為空
                    max_angle = round(max(self.datas[i]), 2)
                    min_angle = round(min(self.datas[i]), 2)
                    avg_angle = round(sta.mean(self.datas[i]), 2)

                    print(f'{self.angleMap[i]}\t最大角度:{max_angle}\t最小角度:{min_angle}\t平均角度:{avg_angle}')

                    # 將結果存入資料庫
                    if self.cursor and self.conn:
                        self.cursor.execute("""
                            INSERT INTO video_result_data (timestamp, pose_id, up_date, average_date, down_date)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (current_time, i, max_angle, avg_angle, min_angle))
                        self.conn.commit()  # 提交更改

                else:
                    print(f'{self.angleMap[i]} 沒有足夠的數據來計算。')
        else:
            print("\n左側角度統計：")
            for i in self.leftAnglePoint:
                if self.datas[i]:  # 確認列表不為空
                    max_angle = round(max(self.datas[i]), 2)
                    min_angle = round(min(self.datas[i]), 2)
                    avg_angle = round(sta.mean(self.datas[i]), 2)

                    print(f'{self.angleMap[i]}\t最大角度:{max_angle}\t最小角度:{min_angle}\t平均角度:{avg_angle}')

                    # 將結果存入資料庫
                    if self.cursor and self.conn:
                        self.cursor.execute("""
                            INSERT INTO video_result_data (timestamp, pose_id, up_date, average_date, down_date)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (current_time, i, max_angle, avg_angle, min_angle))
                        self.conn.commit()  # 提交更改

                else:
                    print(f'{self.angleMap[i]} 沒有足夠的數據來計算。')

    # 清除之前的數據
    def clear_data(self):
        for key in self.datas:
            self.datas[key].clear()
    
    # 關閉數據庫連接
    def close_database_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("數據庫連接已關閉")

class PostureCheck:
    def __init__(self):
        self.suggested_angles = {
            'knee_min': (68, 72),
            'knee_max': (140, 145),
            'ankle_min': (70, 80),
            'ankle_max': (95, 105),
            'hip_min': (55, 65)
        }

    def check_angle_in_range(self, angle, range_min, range_max):
        return range_min <= angle <= range_max

    def check_posture(self, angle_data):
        results = {}
        for joint, angle in angle_data.items():
            suggested_min, suggested_max = self.suggested_angles.get(joint, (None, None))
            if suggested_min is not None and suggested_max is not None:
                if self.check_angle_in_range(angle, suggested_min, suggested_max):
                    results[joint] = f"{joint} 符合建議範圍"
                else:
                    results[joint] = f"{joint} 不符合建議範圍"
            else:
                results[joint] = f"{joint} 沒有建議範圍"
        return results

if __name__ == '__main__':
    posture_checker = PostureCheck()
    tmp = OpticalFlowBikeFit()  # 使用光流追蹤版本

    # 開始處理影像並計算角度
    cap = cv2.VideoCapture("C:\\Users\\User\\Desktop\\IMG_1891.MOV")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # 呼叫 mpPose 來處理影像，儲存角度數據
    output_file = "optical_flow_analysis.mp4"
    print(f"處理影片中，按 q 鍵可提前結束。輸出檔案: {output_file}")
    tmp.mpPose(cap, output_path=output_file)

    # 釋放攝像頭資源
    cap.release()
    
    # 關閉數據庫連接
    tmp.close_database_connection()
    
    print("程式執行完成！")