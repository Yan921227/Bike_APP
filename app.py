#主要code
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files or 'analysis_type' not in request.form:
        return jsonify({'error': 'No file or analysis type provided'}), 400
    
    file = request.files['image']
    analysis_type = request.form['analysis_type']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 處理分析邏輯，例如：
        analysis_result = f"Image received for {analysis_type} analysis."

        print(f"Received file: {filename}")
        print(f"Analysis type: {analysis_type}")
        
        return jsonify({'message': 'File uploaded successfully', 'analysis_result': analysis_result})

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
