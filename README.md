# 🪞 Mirror: Vision-to-Robot Pipeline

> **Extract 3D trajectories from video and deploy directly to Vultr Cloud Robotics.**

A powerful computer vision tool built for the **LabLab AI Hackathon**. "Mirror" takes a video of a human hand gesture (or any tracked object), extracts the 3D motion path, and converts it into executable robot code.

![Screen Shot](https://via.placeholder.com/800x450?text=Mirror+App+Screenshot+Here)

## 🚀 Features

-   **Multi-Mode Tracking**:
    -   ✋ **Hand Tracking**: Uses MediaPipe to track the Index Finger Tip with high precision.
    -   📦 **Object Tracking**: Draw a box around *any* object to track it using OpenCV CSRT.
-   **3D Visualization**: Real-time 3D plotting of the motion path.
-   **Path Smoothing**: Built-in Moving Average filter to create production-ready robot trajectories.
-   **Visual Debugging**: View the "Digital Twin" video with tracking overlays to verify accuracy.
-   **Cloud Deployment**: Simulated one-click deployment to **Vultr Kubernetes Engine**.

## 🛠️ Technology Stack

-   **Frontend**: Streamlit
-   **Computer Vision**: OpenCV, MediaPipe
-   **Data Processing**: NumPy, SciPy
-   **Visualization**: Matplotlib (3D)
-   **Infrastructure**: Vultr ( Simulated Integration)

## 📦 Installation

1.  Clone the repo:
    ```bash
    git clone https://github.com/rahulrocksn/labslablsai-hackathon.git
    cd labslablsai-hackathon
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the app:
    ```bash
    streamlit run app.py
    ```

## 🎮 How to Use

1.  **Upload Video**: Drag and drop an `.mp4` or `.mov` file.
2.  **Select Mode**:
    *   Choose **"Hand Tracking"** for gestures.
    *   Choose **"Manual Object Tracking"** to track specific items (draw a box on the first frame).
3.  **Visual Verification**: Check the "Tracked Overlay" tab to see what the AI is seeing.
4.  **Deploy**: Click "Deploy to Vultr" to push your robot code to the cloud!

---

*Built with ❤️ for the LabLab AI Hackathon*