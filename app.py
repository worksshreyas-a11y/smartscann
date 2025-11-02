import streamlit as st

# ---- Safe OpenCV import for Streamlit Cloud ----
try:
    import cv2
except Exception as e:
    st.error("⚠️ OpenCV import failed: " + str(e))
    st.stop()

import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from matplotlib import pyplot as plt
from streamlit_image_comparison import image_comparison
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import time


# --- PAGE CONFIG ---
st.set_page_config(page_title="SmartScan - AI Object Analyzer", layout="wide")

# --- STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;900&display=swap');
    body {
        background-color: #f8f9fa;
        color: #2c3e50;
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        font-size: 56px;
        text-align: center;
        font-weight: 900;
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        -webkit-background-clip: text;
        color: transparent;
        margin-top: 10px;
    }
    .subtitle {
        text-align: center;
        color: #5a5a5a;
        font-size: 18px;
        margin-bottom: 25px;
    }
    /* --- Warning Box --- */
    .ai-warning {
        background: linear-gradient(135deg, #fff3f3, #ffe0e0);
        border: 2px solid #ff6b6b;
        border-radius: 12px;
        padding: 20px;
        margin: 25px auto 35px;
        max-width: 950px;
        font-size: 16px;
        line-height: 1.7;
        color: #3a0e0e;
        font-weight: 500;
        text-align: justify;
        box-shadow: 0px 4px 15px rgba(255, 99, 99, 0.25);
    }
    .ai-warning b {
        color: #b30000;
        font-weight: 700;
    }
    .ai-warning span {
        font-size: 28px;
        margin-right: 10px;
        vertical-align: middle;
    }
    /* --- Knowledge Note (Old Design Restored) --- */
    .knowledge-note {
        background-color: #e8f7ff;
        border-left: 6px solid #0077b6;
        padding: 15px 20px;
        border-radius: 10px;
        color: #00334d;
        margin-top: 30px;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0px 3px 12px rgba(0, 119, 182, 0.15);
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b4d8;
        color: white;
    }
    footer {
        text-align: center;
        color: #777;
        font-size: 14px;
        margin-top: 50px;
        padding-top: 12px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.markdown('<div class="main-title">🤖 SmartScan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Object Detection, Analysis & Knowledge System</div>', unsafe_allow_html=True)

# --- AI LIMITATION WARNING ---
st.markdown("""
<div class="ai-warning">
<span>⚠️</span> <b>Important Notice:</b><br><br>
SmartScan uses advanced AI and pretrained open-source models (YOLOv8x).  
While highly accurate, results may vary depending on lighting, object clarity, and model bias.  
Always review detected results before use in critical scenarios.
</div>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO("yolov8x.pt")

model = load_model()

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ SmartScan Settings")
    st.markdown("Fine-tune your detection accuracy.")
    conf = st.slider("Confidence Threshold", 0.05, 0.99, 0.45, 0.05)
    iou = st.slider("IOU Threshold (NMS)", 0.05, 0.99, 0.65, 0.05)
    image_title = st.text_input("Image Title / Description", "Scene Analysis")
    st.info("Lower confidence → more detections. Higher confidence → more precision.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    start_time = time.time()
    with st.spinner("🔍 Analyzing image with SmartScan AI..."):
        results = model.predict(img_cv, conf=conf, iou=iou, verbose=False)
    process_time = time.time() - start_time

    result = results[0]
    res_img = result.plot(line_width=2, labels=True, conf=True)
    res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    names = model.names
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    detected_objects = [(names[cls_ids[i]], confs[i], xyxy[i]) for i in range(len(cls_ids))]

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📸 Detection Results", "📊 Analysis & Charts", "📘 Knowledge Base"])

    # --- TAB 1 ---
    with tab1:
        st.markdown(f"### 🖼️ Object Detection for *{image_title}*")
        image_comparison(
            img1=image,
            img2=res_rgb,
            label1="Original Image",
            label2="SmartScan Detection",
            width=800
        )
        st.success(f"✅ {len(detected_objects)} objects detected successfully in {process_time:.2f} seconds.")

    # --- TAB 2 ---
    with tab2:
        if detected_objects:
            obj_names = [o for o, _, _ in detected_objects]
            conf_vals = [c for _, c, _ in detected_objects]
            counts = Counter(obj_names)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 📈 Object Distribution Chart")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(counts.keys(), counts.values(), color="#0077b6", alpha=0.85)
                ax.set_title("Detected Objects per Category", fontsize=13, fontweight='bold')
                ax.set_ylabel("Count")
                ax.set_xlabel("Object Type")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                with st.expander("📊 View Insights"):
                    st.write("This bar chart shows how many instances of each object type were detected in the image. Taller bars indicate more frequent objects.")

            with col2:
                st.metric("Total Objects", len(detected_objects))
                st.metric("Unique Classes", len(counts))
                st.metric("Avg Confidence", f"{np.mean(conf_vals):.3f}")
                with st.expander("📊 View Insights"):
                    st.write("These metrics summarize detection performance — total detected items, unique object categories, and the average confidence score across all detections.")

            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.hist(conf_vals, bins=10, color="#00b4d8", alpha=0.8)
            ax2.set_title("Confidence Score Distribution", fontsize=13, fontweight='bold')
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)
            with st.expander("📊 View Insights"):
                st.write("This histogram shows how confident the AI model was for each detected object. A higher concentration towards 1.0 means more reliable detections.")

            report_data = pd.DataFrame([{
                "Object": obj.title(),
                "Confidence": f"{conf:.3f}",
                "Coordinates": str(coords.tolist())
            } for obj, conf, coords in detected_objects])
            st.dataframe(report_data, use_container_width=True)
            with st.expander("📊 View Insights"):
                st.write("This table lists each detected object with its confidence score and bounding box coordinates for detailed positional analysis.")
            st.download_button("⬇️ Download Detection Report", report_data.to_csv(index=False).encode('utf-8'),
                               file_name="SmartScan_Report.csv", mime="text/csv")

    # --- TAB 3 ---
    with tab3:
        st.markdown("### 📘 Object Knowledge")

        if detected_objects:
            unique_objs = sorted(set([o for o, _, _ in detected_objects]))
            for obj in unique_objs:
                st.subheader(f"🔹 {obj.title()}")
                try:
                    summary = wikipedia.summary(obj, sentences=2)
                    st.write(summary)
                    st.caption("📘 Source: Wikipedia")
                except Exception:
                    st.write(f"SmartScan couldn't find Wikipedia info for '{obj.title()}'.")
                    st.caption("💡 Source: SmartScan Knowledge Engine")

            # --- RESTORED KNOWLEDGE NOTE (Old Blue Design) ---
            st.markdown("""
            <div class="knowledge-note">
            🧠 <b>Note:</b> Some detected objects may not appear on Wikipedia or may show limited/inaccurate descriptions.  
            SmartScan relies on open-source knowledge and pretrained datasets, which may cause occasional mismatches.
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<footer>
Developed by <b>Shreyas Shree</b> | SmartScan © 2025 | AI Object Recognition & Knowledge Platform
</footer>
""", unsafe_allow_html=True)

