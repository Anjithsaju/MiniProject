import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Initialize the pyttsx3 engine once
engine = pyttsx3.init()

# Function for text-to-speech
def speech(text):
    engine.say(text)
    engine.runAndWait()

# Load the YOLO model
model = YOLO('./neck2.pt')  # Replace with the path to your trained model

# Open the webcam (Use 0 for the default webcam, or 1, 2 for external cameras)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# Create a resizable window
cv2.namedWindow('YOLOv8 Real-Time Detection', cv2.WINDOW_NORMAL)
cv2.moveWindow('YOLOv8 Real-Time Detection', 100, 100)
cv2.resizeWindow('YOLOv8 Real-Time Detection', 800, 600)

# Timer to control audio feedback
last_audio_time = 0

# Mapping classes to spoken feedback
class_to_text = {
    "500_new": "500 rupee note detected",
    "500_folded": "500 rupee note detected",
    "200_new": "200 rupee note detected",
    "200_new_folded": "200 rupee note detected",
    "100_new": "100 rupee note detected",
    "100_new_folded": "100 rupee note detected",
    "50_new": "50 rupee note detected",
}

totalsum = 0
seen_objects = set()
counting_active = False

while True:
    ret, frame = cap.read()  # Capture frame-by-frame from webcam
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert BGR to RGB (optional for model compatibility)
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use YOLOv8 model to predict objects in the frame
    results = model.track(frame, persist=True, conf=0.75, verbose=False)

    current_time = time.time()

    # Terminate counting mode if no audio feedback for 10 seconds
    if counting_active and (current_time - last_audio_time >= 10):
        print("Counting mode terminated due to inactivity\n")
        speech(f"Counting mode terminated, total sum is {totalsum} rupees")
        totalsum = 0
        counting_active = False
        seen_objects.clear()
        print("Cleared seen objects")

    # Annotate the frame with YOLO predictions
    if results:
        annotated_frame = results[0].plot()  # Visualize predictions

        for box in results[0].boxes:
            class_id = int(box.cls)
            try:
                object_id = int(box.id)
            except:
                print("int issues")
                continue

            class_name = model.names[class_id]
            unique_object = (object_id, class_name)

            if unique_object not in seen_objects:
                if class_name in class_to_text:
                    feedback_text = class_to_text[class_name]
                    print(feedback_text)
                    speech(feedback_text)

                    content = class_name.split('_')
                    rupees = int(content[0])
                    checker = class_name[-1]

                    if checker != "folded":
                        if not counting_active:
                            print("New counting mode started")
                            speech("New counting mode started")
                            counting_active = True
                        totalsum += rupees
                        print(f"Total sum: {totalsum}\n")
                    else:
                        print("Folded note detected")

                    seen_objects.add(unique_object)
                    last_audio_time = current_time
    else:
        annotated_frame = frame

    # Show the annotated frame
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
