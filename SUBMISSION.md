# Submission Kit: Mirror - Vision-to-Robot Pipeline

> **Ready-to-copy content for your LabLab AI submission form!**

---

### 1. Project Title
**Mirror: Vision-to-Robot Pipeline**

### 2. Short Description (One-Liner)
**Extract 3D trajectories from video and deploy directly to Vultr Cloud Robotics.**

### 3. Long Description
"Mirror" is an AI-powered robotics pipeline that bridges the gap between human intuition and machine execution. Traditional robot programming is tedious, requiring complex code and manual waypoint setting. Mirror changes this by allowing anyone to simply "demonstrate" a movement—using their hand or any object—in front of a webcam.

Our application uses **Google MediaPipe** and **OpenCV** to extract precise 3D motion paths from video. This raw data is smoothed using digital signal processing and then analyzed by **Google Gemini 2.0 Flash**, acting as an "AI Safety Supervisor." Gemini reviews the trajectory for safety, translates the movement into natural language, and suggests cloud optimization strategies. Finally, the approved path is converted into Python robot code and prepared for simulated deployment to **Vultr Kubernetes Engine**.

**Key Innovation**: The integration of an LLM (Gemini) as a safety layer in the control loop, ensuring that only validated, smooth, and safe commands reach the physical hardware.

### 4. Technology Tags
`Computer Vision`, `Robotics`, `Google Gemini`, `Python`, `Streamlit`, `Vultr`, `OpenCV`, `MediaPipe`

### 5. Category
`Robotics & Hardware`, `Computer Vision`, `Generative AI`

### 6. GitHub Repository
[https://github.com/rahulrocksn/labslablsai-hackathon](https://github.com/rahulrocksn/labslablsai-hackathon)

### 7. Application URL (Demo)
*Run locally or paste your deployed Streamlit Cloud link here if applicable.*
*(For submission, you can mention it is a "Local Prototype" if not hosted publicly yet)*

---

### 📷 Usage Instructions for Assets

**Cover Image Idea**:
Take a screenshot of the app showing:
1.  The "Tracked Overlay" video with the yellow finger dot.
2.  The 3D Trajectory Plot (blue line).
3.  The Gemini Analysis text.
*Combine these into one collage for a high-impact cover.*

**Video Presentation Script Outline (1-2 Mins)**:
1.  **Hook**: "Programming robots is hard. What if you could just show them what to do?"
2.  **Demo**: Show yourself waving a hand -> Show the 3D Plot updating -> Show the Gemini Analysis ("Safe movement detected").
3.  **Tech**: "We use MediaPipe for tracking and Gemini 1.5 Pro to valid safety and intent."
4.  **Closing**: "This is Mirror. Deploying safe robot logic to the edge with Vultr."
