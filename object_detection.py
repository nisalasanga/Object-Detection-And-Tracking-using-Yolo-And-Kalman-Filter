import cv2
from ultralytics import YOLO
from KalmanFilter import KalmanFilter

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "2.mp4"
cap = cv2.VideoCapture(video_path)



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        for result in results:                                                  # iterate results
            boxes = result.boxes.cpu().numpy()                                  # get boxes on cpu in numpy
            for box in boxes:
                if result.names[int(box.cls[0])] == 'car':                      # iterate boxes
                    (x1,y1,x2,y2) = box.xyxy[0].astype(int)                     # get corner points as int                                             # print boxes
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)        # draw boxes on img

                
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
