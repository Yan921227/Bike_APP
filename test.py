#目前版本備份11/13
#主要code 連接app

import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import tempfile

app = Flask(__name__, static_url_path='/outputs', static_folder='outputs')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
STATIC_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

mp_pose = mp.solutions.pose

class BikeFit():
    def __init__(self):
        self.pose = mp_pose.Pose()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            return {"error": "Failed to open video file"}

        angles = []
        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                elbow_angle = self.calculate_angle(landmarks[13], landmarks[15], landmarks[23])
                hip_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                knee_angle = self.calculate_angle(landmarks[25], landmarks[27], landmarks[29])
                ankle_angle = self.calculate_angle(landmarks[27], landmarks[29], landmarks[31])

                angles.append({
                    "shoulder_angle": shoulder_angle,
                    "elbow_angle": elbow_angle,
                    "hip_angle": hip_angle,
                    "knee_angle": knee_angle,
                    "ankle_angle": ankle_angle
                })

            frame_counter += 1
            if frame_counter % 300 == 0:
                print(f"Processed {frame_counter} frames.")

        cap.release()
        return angles

    def calculate_angle(self, point1, point2, point3):
        vector_a = (point1.x - point2.x, point1.y - point2.y)
        vector_b = (point3.x - point2.x, point3.y - point2.y)
        dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        magnitude_a = math.sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)
        magnitude_b = math.sqrt(vector_b[0] ** 2 + vector_b[1] ** 2)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        angle_radians = math.acos(dot_product / (magnitude_a * magnitude_b))
        return math.degrees(angle_radians)

    def perform_2d_analysis(self, video_path):
        angles = self.process_video(video_path)
        results = {
            "shoulder": calculate_position_statistics(angles, "shoulder"),
            "elbow": calculate_position_statistics(angles, "elbow"),
            "hip": calculate_position_statistics(angles, "hip"),
            "knee": calculate_position_statistics(angles, "knee"),
            "ankle": calculate_position_statistics(angles, "ankle"),
        }
        comparison = {
            "shoulder_correct": 50 <= results['shoulder']['avg'] <= 70,
            "elbow_correct": 150 <= results['elbow']['avg'] <= 160,
            "hip_correct": 35 <= results['hip']['avg'] <= 40,
            "knee_correct": 35 <= results['knee']['avg'] <= 40,
            "ankle_correct": 15 <= results['ankle']['avg'] <= 30
        }
        return {"result": results, "comparison": comparison}

    def process_video_3d(self, video_path, output_video_path):
        plt.switch_backend('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 設定視角（仰角、方位角）
        ax.view_init(elev=90, azim=-90)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            ax.clear()
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_x = [landmarks[i].x for i in [15, 13, 11, 23, 25, 27]]
                left_y = [landmarks[i].y for i in [15, 13, 11, 23, 25, 27]]
                left_z = [landmarks[i].z for i in [15, 13, 11, 23, 25, 27]]

                right_x = [landmarks[i].x for i in [16, 14, 12, 24, 26, 28]]
                right_y = [landmarks[i].y for i in [16, 14, 12, 24, 26, 28]]
                right_z = [landmarks[i].z for i in [16, 14, 12, 24, 26, 28]]

                ax.plot(left_x, left_y, left_z, color='r')
                ax.plot(right_x, right_y, right_z, color='b')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # 垂直翻轉影片
                flipped_image = cv2.flip(image,0 )
                out.write(cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()
        plt.close(fig)

def calculate_position_statistics(angles, part_name):
    max_angle = max([frame[f"{part_name}_angle"] for frame in angles])
    min_angle = min([frame[f"{part_name}_angle"] for frame in angles])
    avg_angle = sum([frame[f"{part_name}_angle"] for frame in angles]) / len(angles)
    return {"max": max_angle, "min": min_angle, "avg": avg_angle}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'media' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['media']
    analysis_type = request.form.get('analysis_type')
    filename = tempfile.mktemp(suffix='.mp4', dir=UPLOAD_FOLDER)
    file.save(filename)

    bike_fit = BikeFit()

    if analysis_type == "3D":
        output_video_path = os.path.join(OUTPUT_FOLDER, "3d_analysis_output.mp4")
        bike_fit.process_video_3d(filename, output_video_path)
        
        # 返回影片的 URL 給前端
        video_url = url_for('static', filename='3d_analysis_output.mp4', _external=True)
        return jsonify({"video_url": video_url})

    elif analysis_type == "2D":
        result = bike_fit.perform_2d_analysis(filename)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
