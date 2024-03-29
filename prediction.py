import os
import cv2
from ultralytics import YOLO

# Function to process frames
def process_frames(cap, model, threshold):
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if it's time to skip frames
        if frame_count % 5 != 0:
            frame_count += 1
            continue

        frame_count += 1

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        height, width, _ = frame.shape
        scale_factor = 640 / width
        resized_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor) + 50))
        cv2.imshow('Video Stream', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main function
def main():
    cap = cv2.VideoCapture(0)
    model_path = "C:\\Users\\PTC\\Desktop\\Machine Learning yolo\\Yawning detection\\runs\\detect\\train\\weights\\last.pt"
    model = YOLO(model_path)
    threshold = 0.70

    process_frames(cap, model, threshold)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
