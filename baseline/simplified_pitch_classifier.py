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
            'horizontal_movement': abs(x[-1] - x[0]),
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'curvature': curvature,
            'curve_ratio': curve_ratio,
            'total_travel': total_distance
        }
        
        return features
    
    except Exception as e:
        return None

def predict_pitch_type(features):
    if not features:
        return "unknown"
    
    curvature = features['curvature']
    horizontal_movement = features['horizontal_movement']
    vertical_drop = features['vertical_drop']
    curve_ratio = features['curve_ratio']
    avg_velocity = features['avg_velocity']
    
    if curvature > 0.005 and vertical_drop > 80:
        pitch_type = "curveball"
    elif curvature > 0.002 and horizontal_movement > 70:
        pitch_type = "slider"
    elif avg_velocity < 12 and curve_ratio < 1.15:
        pitch_type = "changeup"
    elif vertical_drop > 60 and horizontal_movement > 40 and curvature < 0.003:
        pitch_type = "sinker"
    elif curvature > 0.004 and vertical_drop > 60 and horizontal_movement > 50:
        pitch_type = "knucklecurve"
    else:
        pitch_type = "fastball"
    
    return pitch_type

def map_true_pitch_to_binary(pitch_type):
    if pitch_type.lower() in ["fastball", "sinker"]:
        return "fastball"
    else:
        return "offspeed"

def map_predicted_pitch_to_binary(pitch_type):
    if pitch_type.lower() == "fastball":
        return "fastball"
    else:
        return "offspeed"

def analyze_pitch(video_path):
    ball_positions = process_video(video_path)
    
    if ball_positions is not None and len(ball_positions) > 0:
        features = extract_features(ball_positions)
        
        if features:
            detailed_type = predict_pitch_type(features)
            binary_class = map_predicted_pitch_to_binary(detailed_type)
            
            return detailed_type, binary_class
    
    return "unknown", "unknown"

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
    data_dir = "../baseline_data"
    json_path = "../data/mlb-youtube-segmented.json"
    
    pitch_data = load_pitch_data(json_path)
    
    if not pitch_data:
        return
    
    video_files = [f for f in os.listdir(data_dir) if f.endswith(('.mp4', '.mov'))]
    
    if not video_files:
        return
    
    detailed_results = []
    binary_results = []
    detailed_correct = 0
    binary_correct = 0
    total_predictions = 0
    
    for filename in video_files:
        video_path = os.path.join(data_dir, filename)
        video_id = extract_video_id_from_filename(filename)
        
        true_detailed_type = "unknown"
        if video_id in pitch_data:
            if "type" in pitch_data[video_id]:
                true_detailed_type = pitch_data[video_id]["type"]
        
        if true_detailed_type == "unknown":
            continue
        
        true_binary_class = map_true_pitch_to_binary(true_detailed_type)
        
        predicted_detailed_type, predicted_binary_class = analyze_pitch(video_path)
        
        detailed_is_correct = predicted_detailed_type == true_detailed_type
        binary_is_correct = predicted_binary_class == true_binary_class
        
        if detailed_is_correct:
            detailed_correct += 1
        if binary_is_correct:
            binary_correct += 1
            
        total_predictions += 1
        
        detailed_results.append({
            "filename": filename,
            "true_type": true_detailed_type,
            "predicted_type": predicted_detailed_type,
            "correct": detailed_is_correct
        })
        
        binary_results.append({
            "filename": filename,
            "true_type": true_detailed_type,
            "true_binary": true_binary_class,
            "predicted_binary": predicted_binary_class,
            "correct": binary_is_correct
        })
    
    if total_predictions > 0:
        detailed_accuracy = detailed_correct / total_predictions * 100
        binary_accuracy = binary_correct / total_predictions * 100
        
        true_detailed = [r["true_type"] for r in detailed_results]
        pred_detailed = [r["predicted_type"] for r in detailed_results]
        
        true_binary = [r["true_binary"] for r in binary_results]
        pred_binary = [r["predicted_binary"] for r in binary_results]
        
        confusion = {
            "fastball": {"fastball": 0, "offspeed": 0},
            "offspeed": {"fastball": 0, "offspeed": 0}
        }
        
        for r in binary_results:
            true_class = r["true_binary"]
            pred_class = r["predicted_binary"]
            if true_class in confusion and pred_class in confusion[true_class]:
                confusion[true_class][pred_class] += 1
        
        true_pos = confusion['fastball']['fastball']
        false_pos = confusion['offspeed']['fastball']
        true_neg = confusion['offspeed']['offspeed']
        false_neg = confusion['fastball']['offspeed']
        
        precision_fastball = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall_fastball = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1_fastball = 2 * (precision_fastball * recall_fastball) / (precision_fastball + recall_fastball) if (precision_fastball + recall_fastball) > 0 else 0
        
        precision_offspeed = true_neg / (true_neg + false_neg) if (true_neg + false_neg) > 0 else 0
        recall_offspeed = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        f1_offspeed = 2 * (precision_offspeed * recall_offspeed) / (precision_offspeed + recall_offspeed) if (precision_offspeed + recall_offspeed) > 0 else 0

if __name__ == "__main__":
    main()
