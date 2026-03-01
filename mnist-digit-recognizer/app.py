import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit AI", layout="wide")

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
.big-digit {
    font-size: 140px;
    font-weight: bold;
    text-align: center;
}
.conf-text {
    font-size: 20px;
    text-align: center;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = load_model("model/digit_model.h5")

st.title("🧠 Real-Time MNIST Digit Recognition Dashboard")

col1, col2 = st.columns([1, 1])

# ---------------- LEFT PANEL ---------------- #
with col1:
    st.subheader("🎨 Drawing Area")

    brush_size = st.slider("Brush Size", 5, 40, 18)

    # Session state init
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    if "predict_now" not in st.session_state:
        st.session_state.predict_now = False

    # Buttons row
    b1, b2, b3 = st.columns(3)

    # Clear Canvas
    if b1.button("🗑 Clear"):
        st.session_state.canvas_key += 1
        st.session_state.predict_now = False

    # Reset App
    if b2.button("🔄 Reset App"):
        st.session_state.clear()
        st.rerun()

    # Predict Button
    if b3.button("🧠 Predict"):
        st.session_state.predict_now = True

    # Canvas
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=brush_size,
        stroke_color="white",
        background_color="black",
        height=320,
        width=320,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        display_toolbar=True
    )

# ---------------- RIGHT PANEL ---------------- #
with col2:
    st.subheader("🔎 Recognition Result")

    img = None

    # Only predict when button clicked
    if canvas_result.image_data is not None and st.session_state.predict_now:
        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2GRAY)

    if img is None:
        st.info("Draw a digit and click Predict.")
    else:
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Big digit display
        st.markdown(
            f"<div class='big-digit'>{predicted_digit}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='conf-text'>Confidence: {confidence*100:.2f}%</div>",
            unsafe_allow_html=True
        )

        st.progress(confidence)

        st.markdown("### 🏆 Top 3 Predictions")

        top3 = prediction[0].argsort()[-3:][::-1]

        for i in top3:
            st.write(f"Digit {i} — {prediction[0][i]*100:.2f}%")
            st.progress(float(prediction[0][i]))

        st.markdown("---")
        st.write("📌 Processed 28x28 Preview")
        st.image(img.reshape(28, 28), width=150)

        # Download processed image
        img_bytes = cv2.imencode(
            ".png",
            (img.reshape(28, 28) * 255).astype(np.uint8)
        )[1].tobytes()

        st.download_button(
            label="📥 Download Processed Image",
            data=img_bytes,
            file_name="processed_digit.png",
            mime="image/png"
        )