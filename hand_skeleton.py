import cv2
import mediapipe as mp

# --- THE FIX ---

from mediapipe.python.solutions import hands as mp_hands_solution
from mediapipe.python.solutions import drawing_utils as mp_drawing_solution
from mediapipe.python.solutions import drawing_styles as mp_styles_solution

# Map them to the names the code expects
mp_hands = mp_hands_solution
mp_drawing = mp_drawing_solution
mp_drawing_styles = mp_styles_solution
# ----------------

# 2. Connect to Camera
cap = cv2.VideoCapture(0)

# 3. Start the Hand Tracking
print("Starting Skeleton Tracker... Press 'q' to exit.")

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 4. Process the image

        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        # Draw the skeleton on the frame
        frame.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 5. Show the result
        cv2.imshow('Hand Skeleton', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()