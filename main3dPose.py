#3D+骨架的
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 關閉 TensorFlow 中的一些優化選項，以避免可能的性能問題
import cv2  # OpenCV，用於影像處理
import matplotlib.pyplot as plt  # 用於繪圖
import mediapipe as mp  # Mediapipe，用於姿勢偵測
from mpl_toolkits.mplot3d import Axes3D  # 用於3D繪圖
import math  # 用於數學計算
import time  # 用於計時
import statistics as sta  # 用於統計計算
import tensorflow as tf  # TensorFlow，用於機器學習
# 初始化 TensorFlow 日誌
tf.get_logger().setLevel('INFO')

class BikeFit():
    mp_pose = mp.solutions.pose  # 初始化 Mediapipe 的姿勢偵測模組

    # 定義右側身體關鍵點的索引，這些索引用於繪圖
    rightPoint = [16, 14, 12, 24, 26, 28, 32, 30]
    # 定義左側身體關鍵點的索引，這些索引用於繪圖
    leftPoint = [15, 13, 11, 23, 25, 27, 31, 29]

    # 用於角度計算的右側關鍵點索引
    rightAnglePoint = [12, 14, 24, 26, 28]
    # 用於角度計算的左側關鍵點索引
    leftAnglePoint = [11, 13, 23, 25, 27]

    # 存儲不同關鍵點的數據
    datas = {11: [], 12: [], 13: [], 14: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: []}

    # 關鍵點對應的中文名稱
    angleMap = {11: "肩", 12: "肩",
                13: "手肘", 14: "手肘",
                15: "手腕", 16: "手腕",
                23: "髖", 24: "髖",
                25: "膝", 26: "膝",
                27: "腳踝", 28: "腳踝"}

    # 關鍵點對應的英文名稱
    enAngleMap = {11: "shoulder", 12: "shoulder",
                  13: "elbow", 14: "elbow",
                  15: "wrist", 16: "wrist",
                  23: "hip", 24: "hip",
                  25: "knee", 26: "knee",
                  27: "ankle", 28: "ankle"}

    def __init__(self):
        # 初始化姿勢偵測模組
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils  # 用於繪製姿勢點和骨架

    def process_video(self, video_path):
        # 初始化3D繪圖圖表
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 開啟影片檔案
        cap = cv2.VideoCapture(video_path)
        scale_factor = 1  # 設定放大係數，用於調整圖形比例
        while cap.isOpened():
            ret, frame = cap.read()  # 讀取影片幀
            if not ret:
                break  # 如果影片讀取失敗，退出循環
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 將影像從 BGR 轉換為 RGB
            results = self.pose.process(frame_rgb)  # 使用 Mediapipe 進行姿勢偵測
    
            annotated_frame = self._draw_landmarks(frame, results.pose_landmarks)  # 繪製姿勢點
    
            cv2.imshow('Pose Estimation', annotated_frame)  # 顯示標記後的影像
    
            ax.clear()  # 清除上一幀的繪圖
    
            if results.pose_landmarks:
                # 提取關鍵點的位置
                landmarks = results.pose_landmarks.landmark
                x = [landmark.x * scale_factor for landmark in landmarks]
                y = [landmark.y * scale_factor for landmark in landmarks]
                z = [landmark.z * scale_factor for landmark in landmarks]
    
                # 提取左側關鍵點的位置
                left_x = [landmarks[i].x * scale_factor for i in self.leftPoint]
                left_y = [landmarks[i].y * scale_factor for i in self.leftPoint]
                left_z = [landmarks[i].z * scale_factor for i in self.leftPoint]
    
                # 提取右側關鍵點的位置
                right_x = [landmarks[i].x * scale_factor for i in self.rightPoint]
                right_y = [landmarks[i].y * scale_factor for i in self.rightPoint]
                right_z = [landmarks[i].z * scale_factor for i in self.rightPoint]
                
                # 在3D圖表中繪製左側和右側關鍵點
                ax.scatter(left_x, left_y, left_z, s=10, c='red')
                ax.scatter(right_x, right_y, right_z, s=10, c='blue')
                ax.plot(left_x, left_y, left_z, c='black')
                ax.plot(right_x, right_y, right_z, c='black')

                # 提取髖、膝和腳踝的關鍵點位置
                hip = [landmarks[23].x, landmarks[23].y, landmarks[23].z]  # 髖關節
                knee = [landmarks[25].x, landmarks[25].y, landmarks[25].z]  # 膝關節
                ankle = [landmarks[27].x, landmarks[27].y, landmarks[27].z]  # 腳踝關節
                toe = [landmarks[29].x, landmarks[29].y, landmarks[29].z]  # 腳踝關節
                knee_angle = self.calculate_angle(hip, knee, ankle)  # 計算膝蓋角度
                ankle_angle = self.calculate_angle(knee, ankle, toe)  # 計算腳踝角度
                ax.text2D(0.05, 0.95, f"Knee Angle: {knee_angle:.2f}", transform=ax.transAxes, color='black')  # 在圖表上顯示膝蓋角度
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下 'q' 鍵，退出循環
                    break
            
            # 設置3D圖表的範圍和標籤
            ax.set_xlim(-1 * scale_factor, 1 * scale_factor)
            ax.set_ylim(-1 * scale_factor, 1 * scale_factor)
            ax.set_zlim(-1 * scale_factor, 1 * scale_factor)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
    
            # 顯示圖表
            plt.pause(0.01)
            plt.show(block=False)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下 'q' 鍵，退出循環
                break
        cap.release()  # 釋放影片資源
        cv2.destroyAllWindows()  # 關閉所有 OpenCV 視窗

    def _draw_landmarks(self, image, landmarks):
        # 繪製姿勢關鍵點和骨架
        if landmarks is not None:
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS)
            return annotated_image  # 返回繪製後的影像
        else:
            return image  # 如果沒有檢測到關鍵點，返回原始影像

    def calculate_angle(self, point1, point2, point3):
        # 計算三個點之間的角度
        vector_a = (point1[0] - point2[0], point1[1] - point2[1])  # 計算向量A
        vector_b = (point3[0] - point2[0], point3[1] - point2[1])  # 計算向量B
        dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]  # 計算內積
        magnitude_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)  # 計算向量A的長度
        magnitude_b = math.sqrt(vector_b[0] ** 2 + vector_b[1] ** 2)  # 計算向量B的長度
        if magnitude_a == 0 or magnitude_b == 0:
            return 0  # 如果任何一個向量的長度為0，返回角度為0
        angle_radians = math.acos(dot_product / (magnitude_a * magnitude_b))  # 計算兩向量之間的夾角（弧度）
        angle_degrees = math.degrees(angle_radians)  # 將弧度轉換為角度
        return angle_degrees  # 返回角度

    
if __name__ == '__main__':
    bikeFit = BikeFit()  # 創建 BikeFit 類的實例
    bikeFit.process_video("openCVtest/IMG_6442.MOV")  # 處理指定影片
    cv2.destroyAllWindows()  # 結束所有 OpenCV 視窗
    
    '''
    def mpPose(self,cap):
        mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
        mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
        mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            while True:
                ret, img = cap.read()
                if not ret:
                    print("Cannot receive frame")
                    break
                img = cv2.resize(img,(520,300))               # 縮小尺寸，加快演算速度
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
                results = pose.process(img2)                  # 取得姿勢偵測結果
                # 根據姿勢偵測結果，標記身體節點和骨架
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                cv2.imshow('oxxostudio', img)
                if cv2.waitKey(5) == ord('q'):
                    break     # 按下 q 鍵停止
        cap.release()
        cv2.destroyAllWindows()

'''