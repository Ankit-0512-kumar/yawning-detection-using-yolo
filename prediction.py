# import os
# import cv2
# import threading
# from ultralytics import YOLO

# # Function to process frames in a separate thread
# def process_frames(cap, model, threshold):
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Check if it's time to skip frames
#         if frame_count % 5 != 0:
#             frame_count += 1
#             continue

#         frame_count += 1

#         results = model(frame)[0]

#         for result in results.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = result
#             print(score,"socre=====================")
#             if score > threshold:
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#                 # if class_id == 0 and score > 0.6:
#                 cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#                 # elif class_id == 1:
#                 #     cv2.putText(frame, "Cigarette", (int(x1), int(y1 - 10)),
#                 #                 cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#         height, width, _ = frame.shape
#         scale_factor = 640 / width
#         resized_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor) + 50))
#         cv2.imshow('Video Stream', resized_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Main function
# def main():
#     cap = cv2.VideoCapture(0)
#     # cap = cv2.VideoCapture("rtsp://192.168.0.13:554/avstream/channel=1/stream=1.sdp")
#     # model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
#     # model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
#     model = YOLO("C:\\Users\\PTC\\Desktop\\Machine Learning yolo\\smoking new 1303\\runs\\detect\\train\\weights\\last.pt")
#     threshold = 0.70

#     # "C:\Users\PTC\Desktop\Machine Learning yolo\smoking detection\runs\detect\train\weights\last.pt"

#     # Start a separate thread for processing frames
#     frame_thread = threading.Thread(target=process_frames, args=(cap, model, threshold))
#     frame_thread.start()

#     # Wait for the thread to finish
#     frame_thread.join()

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

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
