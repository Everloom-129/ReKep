from dds_cloudapi_sdk import Config, Client, DetectionTask, TextPrompt, DetectionModel, DetectionTarget
from dds_cloudapi_sdk.tasks.dinox import DinoxTask

import os
import numpy as np
import cv2
import supervision as sv

API_TOKEN = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
MODEL = "GDino1_5_Pro"
DETECTION_TARGETS = ["Mask", "BBox"]

class GroundingDINO:
    def __init__(self):
        config = Config(API_TOKEN)
        self.client = Client(config)

    def detect_objects(self, image_path, input_prompts):
        image_url = self.client.upload_file(image_path)
        task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=pt) for pt in input_prompts],
            targets=[getattr(DetectionTarget, target) for target in DETECTION_TARGETS],
            model=getattr(DetectionModel, MODEL),
        )
        self.client.run_task(task)
        return task.result
    
    def get_dinox(self, image_path, input_prompts):
        TEXT_PROMPT = "<prompt_free>" # TODO: fix as prompt-free mode here
        image_url = self.client.upload_file(image_path)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=TEXT_PROMPT)],
            bbox_threshold=0.25,
            targets=[DetectionTarget.BBox, DetectionTarget.Mask]
        )
        self.client.run_task(task)
        predictions = task.result.objects
        return predictions
    
    def visualize_bbox_and_mask(self, predictions, img_path, output_dir):
        # decode the prediction results
        classes = [pred.category for pred in predictions]
        classes = list(set(classes))
        class_name_to_id = {name: id for id, name in enumerate(classes)}
        class_id_to_name = {id: name for name, id in class_name_to_id.items()}

        boxes = []
        masks = []
        confidences = []
        class_names = []
        class_ids = []

        for idx, obj in enumerate(predictions):
            boxes.append(obj.bbox)
            masks.append(DetectionTask.rle2mask(DetectionTask.string2rle(obj.mask.counts), obj.mask.size))  # convert mask to np.array using DDS API
            confidences.append(obj.score)
            cls_name = obj.category.lower().strip()
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])

        boxes = np.array(boxes)
        masks = np.array(masks)
        class_ids = np.array(class_ids)
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy = boxes,
            mask = masks.astype(bool),
            class_id = class_ids,
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(output_dir, "dinox_bbox.jpg"), annotated_frame)


        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(output_dir, "dinox_mask.jpg"), annotated_frame)
        
        print(f"Annotated image {img_path} has already been saved to {output_dir}")
        print(f"\033[92mDebug: # Boxes: {len(boxes)}\033[0m")
        return boxes, masks


    def rle2rgba(self, rle_mask):
        # Create a dummy task with minimal required arguments
        dummy_task = DetectionTask(
            image_url="dummy",
            prompts=[TextPrompt(text="dummy")],
            targets=[DetectionTarget.Mask],
            model=getattr(DetectionModel, MODEL)
        )
        return dummy_task.rle2rgba(rle_mask)
    