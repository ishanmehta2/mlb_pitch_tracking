import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import Counter

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    ball_positions = []
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        fg_mask = bg_subtractor.apply(frame)
        
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        max_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 500:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > max_circularity and circularity > 0.5:
                max_circularity = circularity
                best_ball = contour
        
        if best_ball is not None:
            M = cv2.moments(best_ball)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ball_positions.append((cx, cy, frame_count))
    
    cap.release()
    return np.array(ball_positions) if ball_positions else None

def extract_features(ball_positions):
    if ball_positions is None or len(ball_positions) < 5:
        return None
        
    x = ball_positions[:, 0]
    y = ball_positions[:, 1]
    t = ball_positions[:, 2]
    
    if len(x) > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        velocities = np.sqrt(dx**2 + dy**2) / dt
    else:
        velocities = np.array([0])
    
    try:
        z = np.polyfit(x, y, 2)
        polynomial = np.poly1d(z)
        y_fit = polynomial(x)
        curvature = abs(z[0])
        
        total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        straight_line_distance = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
        curve_ratio = total_distance / straight_line_distance if straight_line_distance > 0 else 1
        
        features = {
            'vertical_drop': y[-1] - y[0],
            'horizontal_movement': x[-1] - x[0],
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'curvature': curvature,
            'curve_ratio': curve_ratio,
            'total_travel': total_distance,
            'final_x': x[-1],
            'final_y': y[-1],
            'start_x': x[0],
            'start_y': y[0]
        }
        
        return features
    
    except Exception as e:
        return None

def predict_pitch_type(features):
    if not features:
        return "unknown"
    
    curvature = features['curvature']
    horizontal_movement = abs(features['horizontal_movement'])
    vertical_drop = features['vertical_drop']
    curve_ratio = features['curve_ratio']
    
    if curvature > 0.005 and vertical_drop > 80:
        return "curveball"
    elif (curvature > 0.002 and horizontal_movement > 70) or (curvature > 0.0015 and horizontal_movement > 100):
        return "slider"
    elif vertical_drop > 60 and horizontal_movement > 40 and curvature < 0.003:
        return "sinker"
    elif features['avg_velocity'] < 12 and curve_ratio < 1.15:
        return "changeup"
    elif curvature > 0.004 and vertical_drop > 60 and horizontal_movement > 50:
        return "knucklecurve"
    else:
        return "fastball"

def analyze_pitch(video_path):
    ball_positions = process_video(video_path)
    
    if ball_positions is not None and len(ball_positions) > 0:
        features = extract_features(ball_positions)
        
        if features:
            pitch_type = predict_pitch_type(features)
            return pitch_type
    
    return "unknown"

def load_pitch_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {}

def extract_video_id_from_filename(filename):
    return os.path.splitext(filename)[0]

def main():
    data_dir = "baseline_data"
    json_path = "data/mlb-youtube-segmented.json"
    
    pitch_data = load_pitch_data(json_path)
    
    if not pitch_data:
        return
    
    video_files = [f for f in os.listdir(data_dir) if f.endswith(('.mp4', '.mov'))]
    
    if not video_files:
        return
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for filename in video_files:
        video_path = os.path.join(data_dir, filename)
        video_id = extract_video_id_from_filename(filename)
        
        true_type = "unknown"
        if video_id in pitch_data:
            if "type" in pitch_data[video_id]:
                true_type = pitch_data[video_id]["type"]
        
        if true_type == "unknown":
            continue
        
        predicted_type = analyze_pitch(video_path)
        
        is_correct = predicted_type == true_type
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        results.append({
            "filename": filename,
            "true_type": true_type,
            "predicted_type": predicted_type,
            "correct": is_correct
        })
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        
        true_labels = [r["true_type"] for r in results]
        pred_labels = [r["predicted_type"] for r in results]
        
        true_counts = Counter(true_labels)
        pred_counts = Counter(pred_labels)
        
        for pitch_type in set(true_labels):
            type_results = [r for r in results if r["true_type"] == pitch_type]
            type_correct = sum(1 for r in type_results if r["correct"])
            type_total = len(type_results)
            type_accuracy = type_correct / type_total * 100 if type_total > 0 else 0

if __name__ == "__main__":
    main()
