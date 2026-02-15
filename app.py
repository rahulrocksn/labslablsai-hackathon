import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tempfile
import time
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Mirror: Vision-to-Robot Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Futuristic Styling ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        color: #0e1117;
        background-color: #00ffcc;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00ccaa;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.5);
        color: #fff;
    }
    
    /* Code Blocks */
    .stCodeBlock {
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #00ffcc;
    }
    
    /* Video border */
    video {
        border: 2px solid #30363d;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("🪞 Mirror: Vision-to-Robot Pipeline")
st.markdown("### Extract 3D hand trajectories and deploy to Vultr Kubernetes Engine.")
st.markdown("---")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Upload a video of a hand movement to generate robot path code.")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
    
    deploy_target = st.selectbox(
        "Deployment Target",
        ["Vultr Kubernetes Engine (US-West)", "Vultr Kubernetes Engine (EU-Central)", "Local Simulation"]
    )
    
    st.markdown("---")
    st.markdown("**System Status**")
    status_text = st.empty()
    status_text.text("🟢 System Ready")

# --- Helper Functions ---

import subprocess

def process_video(video_path):
    """
    Process video using MediaPipe Hands to extract index finger tip coordinates.
    Also generates an annotated video showing the tracking.
    Returns: Tuple (List of (x, y, z) tuples, Path to processed video)
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], None

    # Video properties for writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temp output file for the raw annotated video (pre-ffmpeg)
    tfile_raw = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    raw_path = tfile_raw.name
    tfile_raw.close()
    
    # We use 'mp4v' for OpenCV writing, then convert.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    coordinates = []
    
    # Progress bar reference
    progress_bar = st.progress(0)
    status_log = st.empty()
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = hands.process(image)
        
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        
        # Convert back to BGR for drawing and saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Draw all landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 2. Extract and Highlight Landmark 8 (Index Finger Tip)
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                coordinates.append((index_tip.x, index_tip.y, index_tip.z))
                
                # Get pixel coordinates for drawing
                cx, cy = int(index_tip.x * width), int(index_tip.y * height)
                cv2.circle(image, (cx, cy), 15, (0, 255, 255), -1) # Yellow dot for tip
                
                # Update Status occasionally
                if frame_idx % 10 == 0:
                    status_log.code(f"Frame {frame_idx}/{total_frames}: Tracking Index Tip (x={index_tip.x:.2f}, y={index_tip.y:.2f})")
        
        # Write annotated frame
        out.write(image)
        
        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
        
    cap.release()
    out.release()
    hands.close()
    progress_bar.empty()
    status_log.empty()
    
    # --- FFmpeg Conversion Step ---
    # Convert raw OpenCV output to browser-friendly H.264
    final_output_path = raw_path.replace('.mp4', '_converted.mp4')
    try:
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", raw_path,
            "-vcodec", "libx264", 
            "-pix_fmt", "yuv420p", # Essential for browser compatibility
            "-crf", "23",
            "-preset", "fast",
            final_output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return coordinates, final_output_path
    except Exception as e:
        st.error(f"FFmpeg conversion failed: {e}")
        return coordinates, raw_path # Fallback to raw path if ffmpeg fails

def plot_3d_trajectory(coords):
    """
    Create a matplotlib figure for the 3D trajectory.
    """
    fig = plt.figure(figsize=(8, 6)) # Sizing
    ax = fig.add_subplot(111, projection='3d')
    
    # Style for dark theme
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords] 
    
    # Plot Line
    ax.plot(xs, ys, zs, c='#00ffcc', linewidth=2, label='Robot Path')
    
    # Start and End points
    if coords:
        ax.scatter(xs[0], ys[0], zs[0], c='lime', s=100, label='Start', marker='o')
        ax.scatter(xs[-1], ys[-1], zs[-1], c='red', s=100, label='End', marker='X')
    
    # Labels and Grid
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Depth', color='white')
    
    # Ticks color
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Grid color
    ax.grid(color='#30363d', linestyle='--', linewidth=0.5)
    
    # Remove panes for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # View angle
    ax.view_init(elev=20., azim=-35)
    
    plt.legend(facecolor='#0e1117', labelcolor='white')
    return fig

def generate_robot_code(coords):
    """
    Generate a Python code snippet representing the robot commands.
    """
    code_lines = []
    code_lines.append("import vultr_robotics as vr")
    code_lines.append("import time")
    code_lines.append("")
    code_lines.append("# Initialize connection to Vultr Cloud Controller")
    code_lines.append("robot = vr.RobotController(api_key='vultr_xxx', region='us-west')")
    code_lines.append("robot.connect()")
    code_lines.append("")
    code_lines.append(f"# Trajectory Path ({len(coords)} points)")
    code_lines.append("path_points = [")
    
    # Sample every Nth point to keep code concise
    step = max(1, len(coords) // 50) 
    for i in range(0, len(coords), step):
        p = coords[i]
        # Simulate mapping: normalized (0-1) to millimeters (e.g. 500mm workspace)
        # Inverting Y because screen Y increases downwards
        x_mm = p[0] * 500
        y_mm = (1 - p[1]) * 500 
        z_mm = abs(p[2]) * 1000 # Depth logic varies
        code_lines.append(f"    {{'x': {x_mm:.2f}, 'y': {y_mm:.2f}, 'z': {z_mm:.2f}}},")
        
    code_lines.append("]")
    code_lines.append("")
    code_lines.append("print('Executing path on physical robot...')")
    code_lines.append("robot.move_path(path_points, speed=0.5, interpolation='linear')")
    
    return "\n".join(code_lines)

# --- Main Application Logic ---

from streamlit_drawable_canvas import st_canvas
from PIL import Image

def process_video_object(video_path, initial_bbox):
    """
    Process video using OpenCV Tracker to track a manually selected object.
    initial_bbox: (x, y, w, h) tuple
    Returns: Tuple (List of (x, y, z) tuples, Path to processed video)
    """
    # Create Tracker
    # utilizing CSRT (Channel and Spatial Reliability Tracker) for better accuracy
    tracker = cv2.TrackerCSRT_create()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize tracker with first frame
    ret, frame = cap.read()
    if not ret:
        return [], None
    
    # Init tracker
    tracker.init(frame, initial_bbox)
    
    # Prep Output
    tfile_raw = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    raw_path = tfile_raw.name
    tfile_raw.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
    
    # Write first frame with box
    p1 = (int(initial_bbox[0]), int(initial_bbox[1]))
    p2 = (int(initial_bbox[0] + initial_bbox[2]), int(initial_bbox[1] + initial_bbox[3]))
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    out.write(frame)
    
    coordinates = []
    
    # Add initial point (normalized)
    center_x = (initial_bbox[0] + initial_bbox[2]/2) / width
    center_y = (initial_bbox[1] + initial_bbox[3]/2) / height
    coordinates.append((center_x, center_y, 0.0)) # z=0 for 2D tracking
    
    progress_bar = st.progress(0)
    status_log = st.empty()
    frame_idx = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update tracker
        success, box = tracker.update(frame)
        
        if success:
            (x, y, w, h) = [int(v) for v in box]
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            
            # Center point
            cx = (x + w/2) / width
            cy = (y + h/2) / height
            coordinates.append((cx, cy, 0.0))
            
            # Draw center
            cv2.circle(frame, (int(x + w/2), int(y + h/2)), 5, (0, 0, 255), -1)
            
            if frame_idx % 10 == 0:
                status_log.code(f"Frame {frame_idx}/{total_frames}: Tracking Object (x={cx:.2f}, y={cy:.2f})")
        else:
             status_log.warning(f"Tracking lost at frame {frame_idx}")
             cv2.putText(frame, "Tracking Failure", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
        out.write(frame)
        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            
    cap.release()
    out.release()
    progress_bar.empty()
    status_log.empty()

    # --- FFmpeg Conversion (Reuse logic) ---
    final_output_path = raw_path.replace('.mp4', '_converted.mp4')
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", 
            "-crf", "23", "-preset", "fast", final_output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return coordinates, final_output_path
    except:
        return coordinates, raw_path

def smooth_path(coords, window_size=5):
    """
    Apply a simple moving average filter to smooth the trajectory.
    """
    if len(coords) < window_size:
        return coords
        
    smoothed = []
    # Convert to numpy for easier slicing
    coords_np = np.array(coords)
    
    for i in range(len(coords)):
        # Determine window bounds
        start = max(0, i - window_size // 2)
        end = min(len(coords), i + window_size // 2 + 1)
        
        # Calculate mean of the window
        # axis=0 means mean across the rows (x, y, z)
        window_mean = np.mean(coords_np[start:end], axis=0)
        smoothed.append(tuple(window_mean))
        
    return smoothed

# --- Main Application Logic ---

if uploaded_file is not None:
    # Column Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Video Input Analysis")
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        
        # Mode Selection
        tracking_mode = st.radio("Select Tracking Mode", 
                                 ["Hand Tracking (MediaPipe)", "Manual Object Tracking (Draw Box)"],
                                 horizontal=True)
        
        # State Management
        if 'processed_file_name' not in st.session_state:
            st.session_state.processed_file_name = ""
            st.session_state.coords = []
            st.session_state.tracked_video_path = None
        
        coords = []
        tracked_video_path = None
        
        if tracking_mode == "Hand Tracking (MediaPipe)":
            # Use tabs for Original vs Tracked
            tab1, tab2 = st.tabs(["Original Video", "Tracked Overlay (Debug)"])
            with tab1:
                st.video(tfile.name)
            
            st.markdown("---")
            
            if uploaded_file.name != st.session_state.processed_file_name:
                 status_text.text("🟡 Processing Video...")
                 with st.spinner("Extracting 3D Landmarks..."):
                    try:
                        coords, tracked_video_path = process_video(tfile.name)
                        st.session_state.coords = coords
                        st.session_state.tracked_video_path = tracked_video_path
                        st.session_state.processed_file_name = uploaded_file.name
                        st.success(f"Successfully extracted {len(coords)} path points!")
                        status_text.text("🟢 Video Processed")
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
            else:
                status_text.text("🟢 Video Processed (Cached)")

        else: # Manual Object Tracking
            status_text.text("🔵 Manual Mode Active")
            
            # Extraction logic for Frame 0
            cap = cv2.VideoCapture(tfile.name)
            ret, frame0 = cap.read()
            cap.release()
            
            if ret:
                frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame0_rgb)
                
                st.info("Draw a box around the object to track on the first frame:")
                
                # Canvas
                # Calculate canvas dimensions to fit column
                canvas_width = 400
                scale_factor = canvas_width / img_pil.width
                canvas_height = int(img_pil.height * scale_factor)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#ff0000",
                    background_image=img_pil,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    key="canvas",
                )
                
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if len(objects) > 0:
                        # Get scale-corrected bbox
                        obj = objects[-1] # Take the last drawn object
                        x = int(obj["left"] / scale_factor)
                        y = int(obj["top"] / scale_factor)
                        w = int(obj["width"] / scale_factor)
                        h = int(obj["height"] / scale_factor)
                        
                        bbox = (x, y, w, h)
                        st.write(f"Selected Region: {bbox}")
                        
                        if st.button("Start Tracking Object"):
                            with st.spinner("Tracking Object..."):
                                coords, tracked_video_path = process_video_object(tfile.name, bbox)
                                st.session_state.coords = coords
                                st.session_state.tracked_video_path = tracked_video_path
                                st.session_state.processed_file_name = uploaded_file.name + "_manual"
                                st.success(f"Tracked {len(coords)} frames!")
            else:
                st.error("Could not read video file.")

        # Display Tracked Video (Common Logic)
        if st.session_state.tracked_video_path:
             # Only show this block if the mode matches what we processed, or generally show last result
             st.markdown("#### Tracked Result")
             st.video(st.session_state.tracked_video_path)
             if tracking_mode == "Manual Object Tracking (Draw Box)":
                 st.caption("Green Box = Tracker ROI")
            
    with col2:
        st.subheader("2. 3D Simulation & Code Gen")
        
        # Get Coords
        raw_coords = st.session_state.get('coords', [])
        
        if raw_coords:
            # Smoothing Option
            use_smoothing = st.checkbox("Apply Smoothing Filter (Moving Average)", value=True)
            
            if use_smoothing:
                coords = smooth_path(raw_coords)
                st.info(f"Smoothed path: {len(raw_coords)} points -> Smooth Curve")
            else:
                coords = raw_coords
            
            # 1. 3D Plot
            st.markdown("#### 3D Trajectory Visualization")
            fig = plot_3d_trajectory(coords)
            st.pyplot(fig)
            
            # 2. Code Generation
            st.markdown("#### Generated Robot Control Code")
            robot_code = generate_robot_code(coords)
            st.code(robot_code, language='python')
            
            # 3. Deploy Button
            st.markdown("---")
            if st.button(f"🚀 Deploy to {deploy_target}"):
                with st.spinner("Pushing container to Vultr Container Registry..."):
                    time.sleep(1.5) # Simulate network delay
                
                st.toast(f"Success! Container pushed to {deploy_target}.", icon="✅")
                st.success("Deployment Triggered: Robot is moving!")
                st.balloons()
        else:
            st.info("Waiting for processing to complete...")

else:
    # Landing State
    st.info("👈 Please upload a .mp4 or .mov video file from the sidebar to begin.")
    
    # Placeholder / Demo Content
    col1, col2 = st.columns(2)
    with col1: 
        st.markdown("#### How it works")
        st.markdown("""
        1. **Upload Video**: Hand movements are tracked.
        2. **MediaPipe Tracking**: Extract Index Finger tip (Landmark 8).
        3. **3D Reconstruction**: Mapped to 3D Cartesian space.
        4. **Code Generation**: Python code for robot execution.
        5. **Vultr Deployment**: Push to cloud controller.
        """)
    with col2:
        st.markdown("#### Technologies")
        st.markdown("""
        - **OpenCV & MediaPipe**: Vision Backend
        - **Streamlit**: Application Frontend
        - **Matplotlib**: 3D Visualization
        - **Vultr**: Cloud Infrastructure
        """)

