import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Streamlit Camera/Image Processing", layout="wide")
st.title("📸 Streamlit Camera / URL Image Processing")

# -----------------------------
# Sidebar: Source & Processing
# -----------------------------
source_type = st.sidebar.selectbox("Select source:", ["Webcam", "URL/Image File"])
mode = st.sidebar.selectbox("Processing mode:", ["Normal", "Grayscale", "Binary", "Canny"])

# Parameters
binary_thresh = st.sidebar.slider("Binary threshold", 0, 255, 128)
canny_t1 = st.sidebar.slider("Canny T1", 0, 500, 100)
canny_t2 = st.sidebar.slider("Canny T2", 0, 500, 200)

# URL or file input
url_input = None
uploaded_file = None
if source_type == "URL/Image File":
    url_input = st.text_input("Enter image/video URL or upload file:")
    uploaded_file = st.file_uploader("Or upload image", type=["jpg", "png"])

# -----------------------------
# Utility functions
# -----------------------------
def read_frame(source, cap):
    if source == "Webcam":
        ret, frame = cap.read()
        return ret, frame
    else:
        # URL/file: can be static image
        if uploaded_file and hasattr(uploaded_file, "read"):
            img = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return True, frame
        elif url_input:
            import requests
            from io import BytesIO
            try:
                resp = requests.get(url_input)
                img = Image.open(BytesIO(resp.content))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                return True, frame
            except:
                return False, None
        return False, None

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "Normal":
        return frame, gray
    elif mode == "Grayscale":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray
    elif mode == "Binary":
        _, bin_img = cv2.threshold(gray, binary_thresh, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR), gray
    elif mode == "Canny":
        edges = cv2.Canny(gray, canny_t1, canny_t2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), gray
    else:
        return frame, gray

def draw_hist(gray_img):
    hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    hist_img = np.zeros((120,256,3), np.uint8)
    cv2.normalize(hist, hist, 0, 120, cv2.NORM_MINMAX)
    for x in range(256):
        cv2.line(hist_img,(x,120),(x,120-int(hist[x])),(255,255,255))
    return hist_img

# -----------------------------
# Main Display
# -----------------------------
if source_type == "Webcam":
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    hist_placeholder = st.empty()

    run = st.checkbox("Run Webcam", value=False)

    if run:
        while run:
            ret, frame = read_frame(source_type, cap)
            if not ret:
                st.warning("Cannot read frame from webcam.")
                break

            processed, gray = process_frame(frame)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            hist_img = draw_hist(gray)
            hist_img = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(processed, caption="Processed Output", use_container_width=True)
            hist_placeholder.image(hist_img, caption="Gray Histogram", use_container_width=True)

        cap.release()

else:  # Static image from URL/file
    ret, frame = read_frame(source_type, None)
    if ret:
        processed, gray = process_frame(frame)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        hist_img = draw_hist(gray)
        hist_img = cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)

        st.image(processed, caption="Processed Output", use_container_width=True)
        st.image(hist_img, caption="Gray Histogram", use_container_width=True)
    else:
        st.info("Please provide a valid image URL or upload an image.")
