# Real-Time Hand Detection with YOLOv8

This project demonstrates how to perform real-time hand detection using Python, OpenCV, and a custom-trained YOLOv8 model. It accesses the webcam, identifies hands in the video feed, and draws bounding boxes with a "Yes/No" status indicator.

## 🧠 How It Works

### 1. The AI Model (YOLOv8)
Standard YOLO (You Only Look Once) models are trained on the COCO dataset, which detects general objects like people, cars, and cups. However, standard models do not have a specific label for "Hand."

To solve this, this project uses a **custom fine-tuned model** (`hand_yolov8n.pt`). This model has been specifically trained on thousands of images of human hands.
* **Input:** It takes a single video frame (image).
* **Processing:** It divides the image into a grid and predicts if a hand exists in that grid cell.
* **Output:** It returns coordinates (x, y) for a box around the hand and a confidence score (probability).

### 2. The Code Logic
* **OpenCV:** Captures video from the laptop webcam frame-by-frame.
* **Inference:** Each frame is sent to the YOLO model.
* **Filtering:** We check the detection results. If the class detected is `'hand'` and the confidence is above 50%, we consider it a valid detection.
* **Visualization:** The code draws a green rectangle around the hand and displays "YES (Hand Found)" on the screen. If no hand is seen, it displays "NO".

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RAZAAli901/Hand-Detection.git](https://github.com/RAZAAli901/Hand-Detection.git)
    cd Hand-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install ultralytics opencv-python
    ```

3.  **Run the detection:**
    ```bash
    python Hand-Detection.py
    ```

4.  **Quit:** Press `q` to close the camera window.

## 🚧 Development Journey & Troubleshooting

During the development of this project, we encountered and resolved several specific technical challenges:

* **Model Loading Errors:**
    * *Issue:* Initially, the script failed because it could not locate the custom model file (`hand_yolov8n.pt`).
    * *Fix:* We implemented the `os` library to use absolute paths (`os.path.abspath(__file__)`) ensuring the script always finds the model regardless of the directory it's run from.

* **OpenCV Display Crash (`cv2.imshow`):**
    * *Issue:* The webcam feed would not open, throwing a "size.width > 0" assertion error. This happened because the camera index `0` was occasionally unavailable or locked by another app.
    * *Fix:* Added a robust check `if not cap.isOpened()` to exit gracefully and print a clear error message if the camera fails to initialize.

* **Git Large File Storage (LFS):**
    * *Issue:* The model weights file was too large for standard GitHub pushes, causing the upload to hang.
    * *Fix:* Configured **Git LFS** to track `.pt` files, allowing the large binary model to be uploaded successfully.

## 📂 Files Included
* `Hand-Detection.py`: The main Python script that runs the camera and logic.
* `hand_yolov8n.pt`: The pre-trained AI weights file (the "brain" of the project).

## 🏆 Credits

This project relies on open-source contributions.

* **YOLOv8 Architecture:** Created by [Ultralytics](https://github.com/ultralytics/ultralytics).
* **Hand Detection Model:** The specific `.pt` weights file used here (`hand_yolov8n.pt`) is sourced from **Bingsu**.
    * **Source:** [Hugging Face - Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer)