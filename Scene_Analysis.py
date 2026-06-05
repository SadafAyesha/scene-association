#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import json
from datetime import datetime
import warnings
from supervision.detection.core import Detections
from supervision.annotators.core import BoxAnnotator, LabelAnnotator
warnings.filterwarnings('ignore')

import glob
import os
import yaml

labels_dir = "/home/mssadaf/Documents/Cursor AI Project/Vehicles type.v1i.yolov11/train/labels"
images_dir = "/home/mssadaf/Documents/Cursor AI Project/Vehicles type.v1i.yolov11/train/images"

# Load class names from data.yaml
with open('Vehicles type.v1i.yolov11/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
CLASS_NAMES = data['names']  # ['car', 'motorcycle', 'rickshaw']
CLASS_SET = set(CLASS_NAMES)

def class_name_to_index(name):
    return CLASS_NAMES.index(name)

def filter_and_map_predictions(predictions):
    filtered = []
    for pred in predictions:
        if pred['class_name'] in CLASS_SET:
            pred['class_id'] = class_name_to_index(pred['class_name'])
            filtered.append(pred)
    return filtered

# Get sorted list of image files
image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

# =============================================================================
# SECTION 1: SETUP AND IMPORTS
# =============================================================================

def setup_environment():
    """Setup environment and install required packages"""
    print("🔧 Setting up environment...")
    
    # Install required packages if not already installed
    try:
        from inference.models.yolo_world.yolo_world import YOLOWorld
        print("✅ All required packages are already installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install required packages:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def create_output_directories():
    """Create output directories for results"""
    directories = ['output', 'output/frames', 'output/csv', 'output/visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Output directories created")

# =============================================================================
# SECTION 2: YOLO WORLD MODEL SETUP
# =============================================================================

class YOLOWorldDetector:
    """YOLO World Zero-Shot Object Detection Class"""
    
    def __init__(self, model_size: str = "l", confidence_threshold: float = 0.3):
        """
        Initialize YOLO World detector
        
        Args:
            model_size: Model size ('s', 'm', 'l')
            confidence_threshold: Detection confidence threshold
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.model: Optional[Any] = None
        self.classes: List[str] = []
        
    def load_model(self) -> bool:
        """Load YOLO World model"""
        print(f"🚀 Loading YOLO World {self.model_size.upper()} model...")
        try:
            from inference.models.yolo_world.yolo_world import YOLOWorld
            self.model = YOLOWorld(model_id=f"yolo_world/{self.model_size}")
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def set_traffic_classes(self):
        """Set traffic vehicle classes for detection"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.classes = ["car", "motorcycle", "rickshaw"]
        self.model.set_classes(self.classes)
        print(f"✅ Set detection classes: {self.classes}")
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, detection_results)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Perform inference
        results = self.model.infer(frame, confidence=self.confidence_threshold)
        
        # Convert to supervision detections
        detections = Detections.from_inference(results).with_nms(threshold=0.1)
        
        # Create annotated frame
        annotated_frame = self._annotate_frame(frame, detections)
        
        # Extract detection results
        detection_results = self._extract_detection_data(detections)
        
        return annotated_frame, detection_results
    
    def _annotate_frame(self, frame: np.ndarray, detections: Any) -> np.ndarray:
        """Annotate frame with bounding boxes and labels"""
        annotated_frame = frame.copy()
        
        box_annotator = BoxAnnotator(thickness=2)
        label_annotator = LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections)
        
        return annotated_frame
    
    def _extract_detection_data(self, detections: Any) -> List[Dict]:
        """Extract detection data in structured format"""
        detection_results = []
        
        if not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return detection_results
        
        for i in range(len(detections)):
            if i < len(detections.xyxy):
                bbox = detections.xyxy[i]
                class_id = detections.class_id[i] if hasattr(detections, 'class_id') and i < len(detections.class_id) else 0
                confidence = detections.confidence[i] if hasattr(detections, 'confidence') and i < len(detections.confidence) else 0.0
                
                # Calculate bounding box center
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                detection_data = {
                    'object_id': i,
                    'class_name': self.classes[class_id] if class_id < len(self.classes) else 'unknown',
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'bbox_center': [center_x, center_y],
                    'bbox_size': [width, height],
                    'area': width * height
                }
                
                detection_results.append(detection_data)
        
        return detection_results

# =============================================================================
# SECTION 3: SPATIAL RELATIONSHIP ANALYSIS
# =============================================================================

class SpatialAnalyzer:
    """Spatial relationship analysis between detected objects"""
    
    def __init__(self):
        self.relationship_data: List[pd.DataFrame] = []
    
    def compute_spatial_relationships(self, detections: List[Dict], frame_id: int) -> pd.DataFrame:
        """
        Compute spatial relationships between all detected objects
        
        Args:
            detections: List of detection dictionaries
            frame_id: Current frame ID
            
        Returns:
            DataFrame with spatial relationship data
        """
        relationships = []
        
        for i, obj1 in enumerate(detections):
            for j, obj2 in enumerate(detections):
                if i != j:  # Don't compare object with itself
                    
                    # Calculate Euclidean distance between centers
                    center1 = obj1['bbox_center']
                    center2 = obj2['bbox_center']
                    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    
                    # Calculate relative position vector
                    relative_x = center2[0] - center1[0]
                    relative_y = center2[1] - center1[1]
                    
                    # Determine spatial relationship type
                    relationship_type = self._determine_relationship_type(center1, center2, obj1, obj2)
                    
                    relationship_data = {
                        'frame_id': frame_id,
                        'object1_id': obj1['object_id'],
                        'object1_class': obj1['class_name'],
                        'object1_center': center1,
                        'object2_id': obj2['object_id'],
                        'object2_class': obj2['class_name'],
                        'object2_center': center2,
                        'euclidean_distance': distance,
                        'relative_position_x': relative_x,
                        'relative_position_y': relative_y,
                        'relationship_type': relationship_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    relationships.append(relationship_data)
        
        return pd.DataFrame(relationships)
    
    def _determine_relationship_type(self, center1: List[float], center2: List[float], 
                                   obj1: Dict, obj2: Dict) -> str:
        """Determine the type of spatial relationship between two objects"""
        x1, y1 = center1
        x2, y2 = center2
        
        # Calculate angle between objects
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # Determine relationship based on angle and distance
        if abs(angle) < 45:
            return "right_of"
        elif abs(angle) > 135:
            return "left_of"
        elif 45 <= angle <= 135:
            return "below"
        else:
            return "above"
    
    def create_spatial_matrix(self, detections: List[Dict]) -> np.ndarray:
        """Create spatial relationship matrix for visualization"""
        n_objects = len(detections)
        if n_objects == 0:
            return np.array([])
        
        matrix = np.zeros((n_objects, n_objects))
        
        for i, obj1 in enumerate(detections):
            for j, obj2 in enumerate(detections):
                if i != j:
                    center1 = obj1['bbox_center']
                    center2 = obj2['bbox_center']
                    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                    matrix[i, j] = distance
        
        return matrix

# =============================================================================
# SECTION 4: PERFORMANCE EVALUATION
# =============================================================================

class PerformanceEvaluator:
    """Performance evaluation using IoU and other metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
    
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2] for first bounding box
            bbox2: [x1, y1, x2, y2] for second bounding box
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there is intersection
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Calculate areas
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0

    def evaluate_spatial_accuracy(self, detections: List[Dict], ground_truth: Optional[List[Dict]] = None, ground_truth_dir: Optional[str] = None, frame_id: Optional[int] = None) -> Dict:
        """
        Evaluate spatial accuracy of detections

        Args:
            detections: List of detected objects
            ground_truth: List of ground truth objects (optional, overrides ground_truth_dir)
            ground_truth_dir: Directory containing YOLOv5/YOLOv8 format ground truth labels (optional)
            frame_id: Current frame index (used to load ground truth file)

        Returns:
            Dictionary with evaluation metrics
        """
        # If ground_truth is not provided but ground_truth_dir is, load ground truth for this frame
        if ground_truth is None and ground_truth_dir is not None and frame_id is not None:
            base = os.path.splitext(os.path.basename(image_files[frame_id]))[0]
            label_file = os.path.join(ground_truth_dir, base + ".txt")
            img_file = os.path.join(ground_truth_dir, base + ".jpg")
            
            if not os.path.exists(label_file):
                print(f"No ground truth found for frame {frame_id}")
                return self._calculate_internal_metrics(detections)
                
            # Get image size for converting YOLO coordinates
            if os.path.exists(img_file):
                img = cv2.imread(img_file)
                img_h, img_w = img.shape[:2]
            else:
                img_w, img_h = 1920, 1080  # Default size
            
            ground_truth = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - width/2) * img_w)
                    y1 = int((y_center - height/2) * img_h)
                    x2 = int((x_center + width/2) * img_w)
                    y2 = int((y_center + height/2) * img_h)
                    
                    ground_truth.append({
                        'class_id': class_id,
                        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'unknown',
                        'bbox': [x1, y1, x2, y2]
                    })
            # If you want to convert YOLO bbox to pixel bbox, you need image size here

        if ground_truth is None:
            return self._calculate_internal_metrics(detections)

        # If ground_truth is in YOLO format, convert to pixel bbox using detections' image size
        # Try to infer image size from detections if possible
        if ground_truth and 'bbox_yolo' in ground_truth[0]:
            if detections and 'bbox' in detections[0]:
                # Use first detection's bbox to infer image size
                # This is a hack; ideally, pass image size explicitly
                x1, y1, x2, y2 = detections[0]['bbox']
                img_w = max(x2, x1) * 2  # crude estimate
                img_h = max(y2, y1) * 2
            else:
                img_w, img_h = 1, 1  # fallback
            for gt in ground_truth:
                x, y, w, h = gt['bbox_yolo']
                x1 = int((x - w / 2) * img_w)
                y1 = int((y - h / 2) * img_h)
                x2 = int((x + w / 2) * img_w)
                y2 = int((y + h / 2) * img_h)
                gt['bbox'] = [x1, y1, x2, y2]
                gt['class_name'] = str(gt['class_id'])
            for gt in ground_truth:
                gt.pop('bbox_yolo', None)

        # Initialize metrics
        iou_scores = []
        matched_detections = 0
        matches = []  # Keep track of matched pairs

        # Calculate IoU matrix between all detections and ground truth
        for gt_idx, gt_obj in enumerate(ground_truth):
            for det_idx, det_obj in enumerate(detections):
                if det_obj['class_name'] == gt_obj['class_name']:  # Only compare same class
                    iou = self.calculate_iou(det_obj['bbox'], gt_obj['bbox'])
                    if iou > 0.5:  # IoU threshold
                        matches.append({
                            'gt_idx': gt_idx,
                            'det_idx': det_idx,
                            'iou': iou
                        })
                    iou_scores.append(iou)

        # Find best matches (handle multiple detections of same object)
        matched_gt = set()
        matched_det = set()

        # Sort matches by IoU score
        matches.sort(key=lambda x: x['iou'], reverse=True)

        # Assign matches greedily
        for match in matches:
            if (match['gt_idx'] not in matched_gt and 
                match['det_idx'] not in matched_det):
                matched_gt.add(match['gt_idx'])
                matched_det.add(match['det_idx'])
                matched_detections += 1

        # Calculate metrics with print statements
        total_gt = len(ground_truth)
        total_det = len(detections)
        
        print(f"\nFrame Metrics:")
        print(f"Total Ground Truth Objects: {total_gt}")
        print(f"Total Detected Objects: {total_det}")
        
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        precision = matched_detections / total_det if total_det > 0 else 0.0
        recall = matched_detections / total_gt if total_gt > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Matched Detections: {matched_detections}")
        print(f"Mean IoU: {mean_iou:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1_score:.3f}")
        
        return {
            'mean_iou': mean_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'matched_detections': matched_detections,
            'total_ground_truth': total_gt,
            'total_detections': total_det,
            'false_positives': total_det - matched_detections,
            'false_negatives': total_gt - matched_detections
        }

    def _calculate_internal_metrics(self, detections: List[Dict]) -> Dict:
        """Calculate internal consistency metrics when ground truth is not available"""
        if len(detections) < 2:
            return {'detection_count': len(detections)}
        
        # Calculate average confidence
        confidences = [det['confidence'] for det in detections]
        avg_confidence = np.mean(confidences)
        
        # Calculate spatial distribution metrics
        centers = np.array([det['bbox_center'] for det in detections])
        center_distances = []
        
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                distance = np.sqrt(np.sum((centers[i] - centers[j])**2))
                center_distances.append(distance)
        
        return {
            'detection_count': len(detections),
            'average_confidence': avg_confidence,
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'average_center_distance': np.mean(center_distances) if center_distances else 0.0,
            'spatial_density': len(detections) / (np.mean(center_distances) + 1e-6)
        }

# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================

class Visualizer:
    """Visualization utilities for analysis results"""
    
    def __init__(self):
        self.colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
    
    def visualize_spatial_relationships(self, frame: np.ndarray, detections: List[Dict], 
                                      relationships_df: pd.DataFrame, frame_id: int) -> np.ndarray:
        """Visualize spatial relationships on frame"""
        vis_frame = frame.copy()
        
        # Draw bounding boxes and labels
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            center = det['bbox_center']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
            
            # Add label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw relationship lines
        for _, rel in relationships_df.iterrows():
            if rel['frame_id'] == frame_id:
                center1 = rel['object1_center']
                center2 = rel['object2_center']
                
                # Draw line between objects
                cv2.line(vis_frame, 
                        (int(center1[0]), int(center1[1])), 
                        (int(center2[0]), int(center2[1])), 
                        (0, 0, 255), 2)
                
                # Add distance label
                mid_x = int((center1[0] + center2[0]) / 2)
                mid_y = int((center1[1] + center2[1]) / 2)
                distance_text = f"{rel['euclidean_distance']:.1f}px"
                cv2.putText(vis_frame, distance_text, (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return vis_frame
    
    def plot_spatial_matrix(self, matrix: np.ndarray, class_names: List[str], frame_id: int):
        """Plot spatial relationship matrix as heatmap"""
        if matrix.size == 0:
            print("No objects detected for matrix visualization")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='viridis', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Spatial Relationship Matrix - Frame {frame_id}')
        plt.xlabel('Object 2')
        plt.ylabel('Object 1')
        plt.tight_layout()
        plt.savefig(f'output/visualizations/spatial_matrix_frame_{frame_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_metrics(self, metrics_history: List[Dict]):
        """Plot performance metrics over time"""
        if not metrics_history:
            return
        
        df = pd.DataFrame(metrics_history)
        print("Available columns:", df.columns)  # Debug print
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total detections over time
        if 'total_detections' in df.columns:
            axes[0, 0].plot(df['frame_id'], df['total_detections'])
        elif 'detection_count' in df.columns:
            axes[0, 0].plot(df['frame_id'], df['detection_count'])
        axes[0, 0].set_title('Total Detections Over Time')
        axes[0, 0].set_xlabel('Frame ID')
        axes[0, 0].set_ylabel('Number of Detections')
        
        # IoU over time
        if 'mean_iou' in df.columns:
            axes[0, 1].plot(df['frame_id'], df['mean_iou'])
            axes[0, 1].set_title('Mean IoU Over Time')
            axes[0, 1].set_xlabel('Frame ID')
            axes[0, 1].set_ylabel('IoU')
        else:
            axes[0, 1].set_title('Mean IoU Over Time (Not Available)')
        
        # Precision and Recall
        if 'precision' in df.columns and 'recall' in df.columns:
            axes[1, 0].plot(df['frame_id'], df['precision'], label='Precision')
            axes[1, 0].plot(df['frame_id'], df['recall'], label='Recall')
            axes[1, 0].set_title('Precision and Recall Over Time')
            axes[1, 0].set_xlabel('Frame ID')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
        else:
            axes[1, 0].set_title('Precision and Recall Over Time (Not Available)')
        
        # F1 Score
        if 'f1_score' in df.columns:
            axes[1, 1].plot(df['frame_id'], df['f1_score'])
            axes[1, 1].set_title('F1 Score Over Time')
            axes[1, 1].set_xlabel('Frame ID')
            axes[1, 1].set_ylabel('F1 Score')
        else:
            axes[1, 1].set_title('F1 Score Over Time (Not Available)')
        
        plt.tight_layout()
        plt.savefig('output/visualizations/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

# =============================================================================
# SECTION 6: MAIN PROCESSING PIPELINE
# =============================================================================

class TrafficAnalysisPipeline:
    """Main pipeline for traffic analysis"""
    
    def __init__(self, video_path: str, output_path: str = "output"):
        self.video_path = video_path
        self.output_path = output_path
        self.detector = YOLOWorldDetector()
        self.spatial_analyzer = SpatialAnalyzer()
        self.performance_evaluator = PerformanceEvaluator()
        self.visualizer = Visualizer()
        
        # Data storage
        self.all_relationships: List[pd.DataFrame] = []
        self.performance_metrics: List[Dict] = []
        self.frame_results: List[Dict] = []
        
    def run_analysis(self, max_frames: Optional[int] = None) -> bool:
        """Run complete traffic analysis pipeline"""
        print("🚀 Starting Traffic Analysis Pipeline...")
        
        # Setup
        if not setup_environment():
            return False
        
        create_output_directories()
        
        # Load model
        if not self.detector.load_model():
            return False
        
        self.detector.set_traffic_classes()
        
        # Process video
        self._process_video(max_frames)
        
        # Generate outputs
        self._generate_outputs()
        
        print("✅ Analysis completed successfully!")
        return True
    
    def _process_video(self, max_frames: Optional[int] = None):
        """Process video frames"""
        print("📹 Processing video frames...")
        
        # Load ground truth
        gt_dir = "/home/mssadaf/Documents/Cursor AI Project/Vehicles type.v1i.yolov11/train"
        
        from supervision.utils.video import VideoInfo, get_video_frames_generator
        from tqdm import tqdm
        
        # Get video info
        video_info = VideoInfo.from_video_path(self.video_path)
        print(f"Video info: {video_info.resolution_wh} @ {video_info.fps} FPS")
        
        # Setup video writer
        output_video_path = os.path.join(self.output_path, "traffic_analysis_output.mp4")
        # Use getattr for compatibility with different OpenCV versions
        fourcc_func = getattr(cv2, 'VideoWriter_fourcc', cv2.VideoWriter.fourcc)
        fourcc = fourcc_func(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, video_info.fps, video_info.resolution_wh)
        
        # Process frames
        frame_count = 0
        generator = get_video_frames_generator(self.video_path)
        
        for frame in tqdm(generator, desc="Processing frames"):
            if max_frames and frame_count >= max_frames:
                break
            
            # Detect objects
            annotated_frame, detections = self.detector.detect_objects(frame)
            
            # Filter and map class names to indices
            detections = filter_and_map_predictions(detections)

            # Compute spatial relationships
            relationships_df = self.spatial_analyzer.compute_spatial_relationships(detections, frame_count)
            self.all_relationships.append(relationships_df)
            
            # Evaluate performance with ground truth
            metrics = self.performance_evaluator.evaluate_spatial_accuracy(
                detections=detections,
                ground_truth=None,  # Will be loaded from directory
                ground_truth_dir=gt_dir,
                frame_id=frame_count
            )
            metrics['frame_id'] = frame_count
            self.performance_metrics.append(metrics)
            
            # Create spatial matrix
            spatial_matrix = self.spatial_analyzer.create_spatial_matrix(detections)
            
            # Visualize results
            if len(detections) > 0:
                class_names = [det['class_name'] for det in detections]
                vis_frame = self.visualizer.visualize_spatial_relationships(
                    annotated_frame, detections, relationships_df, frame_count
                )
                
                # Save spatial matrix visualization every 10 frames
                if frame_count % 10 == 0 and spatial_matrix.size > 0:
                    self.visualizer.plot_spatial_matrix(spatial_matrix, class_names, frame_count)
            else:
                vis_frame = annotated_frame
            
            # Write frame
            out.write(vis_frame)
            
            # Save frame results
            self.frame_results.append({
                'frame_id': frame_count,
                'detections': detections,
                'spatial_matrix': spatial_matrix.tolist() if spatial_matrix.size > 0 else []
            })
            
            frame_count += 1
        
        out.release()
        print(f"✅ Processed {frame_count} frames")
    
    def _generate_outputs(self):
        """Generate all output files"""
        print("📊 Generating output files...")
        
        # Combine all relationships
        if self.all_relationships:
            all_relationships_df = pd.concat(self.all_relationships, ignore_index=True)
            all_relationships_df.to_csv(os.path.join(self.output_path, 'csv', 'spatial_relationships.csv'), index=False)
            print("✅ Spatial relationships saved to CSV")
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_df = pd.DataFrame(self.performance_metrics)
            metrics_df.to_csv(os.path.join(self.output_path, 'csv', 'performance_metrics.csv'), index=False)
            print("✅ Performance metrics saved to CSV")
            
            # Save spatial accuracy metrics (mean_iou, precision, recall, f1_score, matched_detections, etc.)
            spatial_accuracy_cols = [
                'frame_id', 'mean_iou', 'precision', 'recall', 'f1_score',
                'matched_detections', 'total_ground_truth', 'total_detections',
                'false_positives', 'false_negatives'
            ]
            spatial_accuracy_df = metrics_df[[col for col in spatial_accuracy_cols if col in metrics_df.columns]]
            spatial_accuracy_df.to_csv(os.path.join(self.output_path, 'csv', 'spatial_accuracy_metrics.csv'), index=False)
            print("✅ Spatial accuracy metrics saved to CSV")
            
            # Plot performance metrics
            self.visualizer.plot_performance_metrics(self.performance_metrics)
            print("✅ Performance metrics visualization saved")
        
        # Save detailed results
        results_summary = {
            'total_frames': len(self.frame_results),
            'total_detections': sum(len(frame['detections']) for frame in self.frame_results),
            'analysis_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_size': self.detector.model_size,
                'confidence_threshold': self.detector.confidence_threshold,
                'classes': self.detector.classes
            }
        }
        
        with open(os.path.join(self.output_path, 'analysis_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("✅ Analysis summary saved")
        print(f"📁 All outputs saved to: {self.output_path}")

# =============================================================================
# SECTION 7: INTERACTIVE MAIN FUNCTION
# =============================================================================

def main():
    """Main interactive function"""
    print("🚗 Interactive Traffic Analysis with YOLO World")
    print("=" * 50)
    
    # Get video path
    video_path = input("Enter video path (or press Enter for default 'traffic.mp4'): ").strip()
    if not video_path:
        video_path = "traffic.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    # Get processing parameters
    try:
        max_frames_input = input("Enter maximum frames to process (or press Enter for all): ").strip()
        max_frames = int(max_frames_input) if max_frames_input else None
    except ValueError:
        print("❌ Invalid number of frames, processing all frames")
        max_frames = None

    # Run analysis
    pipeline = TrafficAnalysisPipeline(video_path)
    success = pipeline.run_analysis(max_frames)
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("\nGenerated outputs:")
        print("- output/traffic_analysis_output.mp4 (annotated video)")
        print("- output/csv/spatial_relationships.csv (spatial relationships)")
        print("- output/csv/performance_metrics.csv (performance metrics)")
        print("- output/csv/spatial_accuracy_metrics.csv (spatial accuracy metrics)")
        print("- output/visualizations/ (visualization plots)")
        print("- output/analysis_summary.json (analysis summary)")
    else:
        print("❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
