#!/usr/bin/env python3

import rospy
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage  # Required for conversion

class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node('object_detector_node')
        self.bridge = CvBridge()

        # Load model with weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        
        # Get class names from metadata
        self.class_names = weights.meta["categories"]
        rospy.loginfo(f"Loaded {len(self.class_names)} classes")

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Use built-in transforms
        self.transform = weights.transforms()

        # ROS setup
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/detected_objects/image', Image, queue_size=1)

    def image_callback(self, msg):
        try:
            # Convert to OpenCV and correct color space
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for transforms
            pil_image = PILImage.fromarray(rgb_image)
        except Exception as e:
            rospy.logerr(f"Image conversion error: {e}")
            return

        # Apply transforms and run model
        img_tensor = self.transform(pil_image).to(self.device)
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        # Process detections
        for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
            if score < 0.5:
                continue
                
            # Convert tensor to int and get class name
            label_idx = label.item()
            
            # Validate class index range
            if 1 <= label_idx <= len(self.class_names):
                class_name = self.class_names[label_idx]  # Offset adjustment
            else:
                rospy.logwarn(f"Invalid class index: {label_idx}")
                continue

            # Draw detection
            x1, y1, x2, y2 = map(int, box.tolist())
            label_text = f"{class_name}: {score:.2f}"

            # Bounding box
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adaptive text size
            scale = max(cv_image.shape[0], cv_image.shape[1]) / 1000
            font_scale = 0.8 * scale
            thickness = max(1, int(1.5 * scale))
            
            # Text background
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(cv_image, 
                          (x1, y1 - text_h - 10),
                          (x1 + text_w, y1),
                          (0, 255, 0), -1)
            
            # Class label
            cv2.putText(cv_image, label_text,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 0, 0), thickness)

        # Publish result
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        except Exception as e:
            rospy.logerr(f"Publish error: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ObjectDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass