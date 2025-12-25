import cv2
from ultralytics import YOLO

# 1. Load the CUSTOM hand model
# Make sure 'hand_yolov8n.pt' is in the same folder as this script!
model = YOLO('hand_yolov8n.pt')

# 2. Connect to Webcam
cap = cv2.VideoCapture(0)

print("Starting Hand Detection... Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # 3. Run detection
    # conf=0.5 means it must be 50% sure it's a hand to say "Yes"
    results = model(frame, verbose=False, conf=0.5)

    hand_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 4. Check class name
            # The model we downloaded detects 'hand', so we look for that.
            # (Sometimes these models label it '0' or 'hand')
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Print what it sees to the console for debugging
            print(f"I see: {class_name}")

            # If it sees a hand, mark it as found
            if 'hand' in class_name.lower():
                hand_detected = True

                # Draw box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 5. Display Status
    if hand_detected:
        status_text = "YES (Hand Found)"
        color = (0, 255, 0)  # Green
    else:
        status_text = "NO"
        color = (0, 0, 255)  # Red

    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()