import cv2
from ultralytics import YOLO
from tracker import Tracker
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "los_angeles.mp4"
cap = cv2.VideoCapture(video_path)

tracker = Tracker(150, 30, 5)
skip_frame_count = 0



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        for result in results:                                                  # iterate results
            boxes = result.boxes.cpu().numpy()                                  # get boxes on cpu in numpy
            centers = []
            for box in boxes:
                if result.names[int(box.cls[0])] == 'car':                      # iterate boxes
                    (x1,y1,x2,y2) = box.xyxy[0].astype(int)                     # get corner points as int                                             # print boxes
                    cx = int((x1 + x2)/2)                                       # Mid point of x 
                    cy = int((y1 + y2)/2)                                       # Mid point of x 
                    cv2.circle(frame,(cx, cy), 5, (0, 0, 255),-1)               # Mid point Drawing
                    centers.append([cx,cy])
                    
                    
            centers = np.array(centers)
            tracker.update(centers)


            for j in range(len(tracker.tracks)):
                if (len(tracker.tracks[j].trace) > 1):
                    x = int(tracker.tracks[j].trace[-1][0,0]) 
                    y = int(tracker.tracks[j].trace[-1][0,1])
                    tl = (x-10,y-10)
                    br = (x+10,y+10)
                    cv2.rectangle(frame,tl,br,(0, 255, 0),1)
                    cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, (0, 255, 0),2)
                    for k in range(len(tracker.tracks[j].trace)):
                        x = int(tracker.tracks[j].trace[-1][0,0])
                        y = int(tracker.tracks[j].trace[-1][0,1])
                        cv2.circle(frame,(x,y), 3, (0, 255, 0),-1)
                        cv2.circle(frame,(x,y), 6, (0, 255, 0),-1)
				
                    
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)        # draw boxes on img
            print(np.array(centers))
                 
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
