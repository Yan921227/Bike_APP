#關節點的版本

import cv2
import mediapipe as mp
import math
import time
import statistics as sta
import mysql.connector




class BikeFIt():
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
        
        # 初始化數據庫連接
        try:
            self.conn = mysql.connector.connect(
                host='localhost',
                user='your_username',
                password='your_password',
                database='your_database'
            )
            self.cursor = self.conn.cursor()
            print("數據庫連接成功")
        except mysql.connector.Error as err:
            print(f"數據庫連接失敗: {err}")
            self.conn = None
            self.cursor = None

    # 找前後元素
    def find_neighboring_elements(self, number, elements_list):
        index = elements_list.index(number)
        previous_element = elements_list[index - 1] if index > 0 else None
        next_element = elements_list[index + 1] if index < len(elements_list) - 1 else None
        return previous_element, next_element

    # 擷取檢測點座標
    def landmarksDetection(self, img, results, draw=False):
        img_height, img_width = img.shape[:2]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      results.pose_landmarks.landmark]
        if draw:
            [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
        return mesh_coord

    # 計算角度的公式
    def calculate_angle(self, point1, point2, point3):
        vector_a = (point1[0] - point2[0], point1[1] - point2[1])
        vector_b = (point3[0] - point2[0], point3[1] - point2[1])
        dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        magnitude_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)
        magnitude_b = math.sqrt(vector_b[0] ** 2 + vector_b[1] ** 2)
        angle_radians = math.acos(dot_product / (magnitude_a * magnitude_b))
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    # 人體姿態檢測 只擷取單側影像
    def mpPose(self, cap, side='right'):
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            if not cap.isOpened():
                print("Cannot open camera")
                exit()

            while True:
                ret, img = cap.read()
                if not ret:
                    print("Cannot receive frame")
                    break

                img = cv2.resize(img, (1080, 720))
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB
                results = pose.process(img2)  # 開始姿勢估計

                if results.pose_landmarks is not None:
                    mesh_coord = self.landmarksDetection(img2, results, True)
                    if side == 'right':
                        [cv2.circle(img, mesh_coord[i], 3, (0, 0, 255), -1) for i in self.rightPoint]
                    else:
                        [cv2.circle(img, mesh_coord[i], 3, (0, 0, 255), -1) for i in self.leftPoint]

                    # 計算角度
                    if side == 'right':
                        for i in self.rightAnglePoint:
                            test1, test2 = self.find_neighboring_elements(i, self.rightPoint)
                            angle = self.calculate_angle(mesh_coord[test1], mesh_coord[i], mesh_coord[test2])
                            self.datas[i].append(angle)
                    else:
                        for i in self.leftAnglePoint:
                            test1, test2 = self.find_neighboring_elements(i, self.leftPoint)
                            angle = self.calculate_angle(mesh_coord[test1], mesh_coord[i], mesh_coord[test2])
                            self.datas[i].append(angle)

                # 每 10 秒統計一次角度
                if time.time() - self.start_time >= self.time_interval:
                    print("\n每 10 秒統計角度：")
                    self.record_angle_statistics(side)
                    self.start_time = time.time()  # 重置開始時間
                    self.clear_data()

                cv2.imshow('Pose', img)

                if cv2.waitKey(1) == ord('q'):
                    break
                

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
    tmp = BikeFIt()

    # 開始處理影像並計算角度
    cap = cv2.VideoCapture("openCVtest/IMG_6442.MOV")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # 呼叫 mpPose 來處理影像，儲存角度數據
    tmp.mpPose(cap)

    # 釋放攝像頭資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()
    
    # 關閉數據庫連接
    tmp.close_database_connection()
