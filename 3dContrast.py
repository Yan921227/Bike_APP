


import os
import cv2
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BikeFitCompare:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        self.leftPoint = [15, 13, 11, 23, 25, 27, 31, 29]
        self.rightPoint = [16, 14, 12, 24, 26, 28, 32, 30]
        self.standard_frames = []  # 標準影片骨架陣列

    def calculate_angle(self, point1, point2, point3):
        vector_a = (point1[0] - point2[0], point1[1] - point2[1])
        vector_b = (point3[0] - point2[0], point3[1] - point2[1])
        dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        magnitude_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)
        magnitude_b = math.sqrt(vector_b[0] ** 2 + vector_b[1] ** 2)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        angle_radians = math.acos(dot_product / (magnitude_a * magnitude_b))
        return math.degrees(angle_radians)

    def extract_pose_sequence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                pose_3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                frames.append(pose_3d)
        cap.release()
        return frames

    def draw_pose_lines(self, ax, pose, indices, color, label=None):
        x = [pose[i][0] for i in indices]
        y = [pose[i][1] for i in indices]
        z = [pose[i][2] for i in indices]
        ax.plot(x, y, z, color=color, label=label if label else "")
        ax.scatter(x, y, z, color=color, s=10)

    def compare_videos(self, user_video_path, standard_video_path):
        print("▶ 正在讀取標準影片骨架資料...")
        self.standard_frames = self.extract_pose_sequence(standard_video_path)

        print("▶ 開始播放與比較使用者影片骨架...")
        cap = cv2.VideoCapture(user_video_path)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_index >= len(self.standard_frames):
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            ax.clear()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                user_pose_3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                self.draw_pose_lines(ax, user_pose_3d, self.leftPoint, 'red', '使用者左')
                self.draw_pose_lines(ax, user_pose_3d, self.rightPoint, 'blue', '使用者右')

                # 膝蓋角度
                hip = [landmarks[23].x, landmarks[23].y]
                knee = [landmarks[25].x, landmarks[25].y]
                ankle = [landmarks[27].x, landmarks[27].y]
                knee_angle = self.calculate_angle(hip, knee, ankle)
                ax.text2D(0.05, 0.95, f"Knee Angle: {knee_angle:.2f}", transform=ax.transAxes)

            # ➕ 畫標準骨架（綠）已做水平翻轉並左移
            standard_pose = self.standard_frames[frame_index].copy()
            standard_pose[:, 0] = -standard_pose[:, 0] - 0.5  # 水平翻轉 + 左移
            self.draw_pose_lines(ax, standard_pose, self.leftPoint, 'green', '標準左')
            self.draw_pose_lines(ax, standard_pose, self.rightPoint, 'green', '標準右')

            # 圖表設定
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title("使用者 vs 標準骨架")
            ax.legend(loc='upper right')
            plt.pause(0.01)

            cv2.imshow('User Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    bikefit = BikeFitCompare()
    bikefit.compare_videos("openCVtest/IMG_1891.MOV", "openCVtest/IMG_6442.MOV")