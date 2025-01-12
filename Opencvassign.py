import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

parent_labels = ["person", "car"]
sub_labels = ["helmet", "tire", "door"]


def process_frame(frame, parent_labels, sub_labels):

    results = model(frame)
    detections = results[0].boxes

    detected_objects = []
    parent_objects = {}


    for detection in detections:
        class_id = int(detection.cls)
        label = model.names[class_id]
        bbox = detection.xyxy[0].tolist()

        if label in parent_labels:

            obj_id = len(parent_objects) + 1
            parent_objects[obj_id] = {
                "object": label,
                "id": obj_id,
                "bbox": bbox,
                "subobjects": []
            }
        elif label in sub_labels:

            for parent_id, parent_data in parent_objects.items():
                parent_bbox = parent_data["bbox"]

                if (
                        parent_bbox[0] <= bbox[0] <= parent_bbox[2] and
                        parent_bbox[1] <= bbox[1] <= parent_bbox[3]
                ):
                    sub_id = len(parent_data["subobjects"]) + 1
                    parent_data["subobjects"].append({
                        "object": label,
                        "id": sub_id,
                        "bbox": bbox
                    })


    for parent_id, parent_data in parent_objects.items():
        detected_objects.append(parent_data)

    return detected_objects


def run_video(video_path, parent_labels, sub_labels):

    cap = cv2.VideoCapture(video_path)

    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        detected_objects = process_frame(frame, parent_labels, sub_labels)


        for obj in detected_objects:

            bbox = list(map(int, obj["bbox"]))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{obj["object"]} {obj["id"]}',
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            for sub in obj["subobjects"]:
                sub_bbox = list(map(int, sub["bbox"]))
                cv2.rectangle(frame, (sub_bbox[0], sub_bbox[1]), (sub_bbox[2], sub_bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, f'{sub["object"]} {sub["id"]}',
                            (sub_bbox[0], sub_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    video_path = "sample1.mp4"

    run_video(video_path, parent_labels, sub_labels)