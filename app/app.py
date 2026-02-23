# debug_app.py — paste/replace into app/app.py
# Shows detailed preprocessing steps and auto/manual inversion

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import os, time, cv2

st.set_page_config(page_title="Digit Recognition Debug", layout="centered")
st.title("🖊️ Digit Recognition ")
st.markdown("This page shows intermediate preprocessing steps so you can see exactly what the model receives.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/emnist_digit_recognizer.h5")

model = load_model()

# canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=250,
    width=250,
    drawing_mode="freedraw",
    key="canvas",
)

# controls
st.sidebar.markdown("## Controls")
auto_invert = st.sidebar.checkbox("Auto invert (brightness check)", value=True)
manual_invert = st.sidebar.checkbox("Force invert manually", value=False)
otsu_threshold = st.sidebar.slider("Threshold (for debug)", min_value=0, max_value=255, value=30)
show_all = st.sidebar.checkbox("Show all intermediate images", value=True)

SAVE_DIR = "user_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def composite_canvas_rgba_to_gray(image_data):
    """
    image_data from streamlit canvas is RGBA numpy array shape (H,W,4).
    Composite alpha onto black background and return grayscale uint8 array.
    """
    # If already single-channel, just return
    if image_data.ndim == 2:
        return image_data.astype("uint8")
    # Separate channels
    rgb = image_data[:, :, :3].astype(np.float32)
    alpha = image_data[:, :, 3].astype(np.float32) / 255.0
    # composite over black: out = rgb * alpha + black*(1-alpha) => rgb*alpha
    comp = (rgb * alpha[:, :, None])
    # convert to grayscale
    gray = comp.mean(axis=2)
    return gray.astype("uint8")

def build_debug_steps(canvas_rgba):
    """Return dict of intermediate PIL images and final model input."""
    steps = {}
    # 1) raw composited grayscale
    gray = composite_canvas_rgba_to_gray(canvas_rgba)
    pil_raw = Image.fromarray(gray).convert("L")
    steps['raw'] = pil_raw

    # 2) resize big to preserve detail
    pil_big = pil_raw.resize((280, 280), Image.NEAREST)
    steps['big'] = pil_big

    # 3) optionally invert based on brightness or manual toggle
    mean_b = np.array(pil_big).mean()
    steps['mean_brightness'] = mean_b

    # We'll decide to invert if mean brightness > 127 (i.e. background mostly white)
    auto_should_invert = mean_b > 127

    # Apply inversion logic externally in caller (we return both)
    pil_inverted = ImageOps.invert(pil_big)
    steps['inverted'] = pil_inverted

    # 4) threshold (use otsu-like if otsu val provided else simple)
    arr_inv = np.array(pil_inverted)
    # adaptive: here use simple threshold from slider (otsu could also be used)
    _, th = cv2.threshold(arr_inv, otsu_threshold, 255, cv2.THRESH_BINARY)
    steps['threshold'] = Image.fromarray(th)

    # 5) bounding box on threshold image
    coords = np.column_stack(np.where(th > 0))
    if coords.size == 0:
        steps['bbox'] = None
        # final blank 28x28
        final28 = Image.fromarray(np.zeros((28,28), dtype=np.uint8))
        steps['final28'] = final28
        return steps

    y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
    # draw crop box as image for debug
    crop_vis = pil_inverted.copy().convert("RGB")
    # crop box preview: draw rectangle (use numpy)
    cv_vis = np.array(crop_vis)
    cv2.rectangle(cv_vis, (x0,y0),(x1,y1),(255,0,0),1)  # red box
    steps['bbox'] = Image.fromarray(cv_vis)

    # 6) crop and downscale to 28x28 while preserving aspect ratio (center in 28x28)
    crop = th[y0:y1+1, x0:x1+1]
    h,w = crop.shape
    # scale to 20 on max side
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    crop_img = Image.fromarray(crop).resize((new_w,new_h), Image.Resampling.LANCZOS)
    new_img = np.zeros((28,28), dtype=np.uint8)
    top = (28-new_h)//2; left = (28-new_w)//2
    new_img[top:top+new_h, left:left+new_w] = np.array(crop_img)
    pil28 = Image.fromarray(new_img).filter(ImageFilter.GaussianBlur(radius=0.5))
    steps['final28'] = pil28

    # final normalized array for model
    final_arr = np.array(pil28).astype(np.float32)/255.0
    # IMPORTANT: model expects white digit on black -> ensure digit is white
    # decide inversion later
    steps['final_arr_pre_invert'] = final_arr  # currently 0..1; digit white=1 or 0 depending prior inversion
    return steps

# Main action
if st.button("Predict / Debug"):
    if canvas_result.image_data is None:
        st.warning("Draw something first")
    else:
        rgba = (canvas_result.image_data).astype("uint8")
        steps = build_debug_steps(rgba)

        # show raw info
        st.write("**Mean brightness (big image):**", float(steps['mean_brightness']))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Raw composited (grayscale):")
            st.image(steps['raw'].resize((200,200)))
        with col2:
            st.write("Resized (big):")
            st.image(steps['big'].resize((200,200)))
        with col3:
            st.write("Inverted (big):")
            st.image(steps['inverted'].resize((200,200)))

        if steps['bbox'] is not None:
            st.write("Bounding box preview (red):")
            st.image(steps['bbox'].resize((200,200)))

        st.write("Thresholded (after inversion):")
        st.image(steps['threshold'].resize((200,200)))

        # decide invert
        should_invert = False
        if manual_invert:
            should_invert = True
        elif auto_invert and steps['mean_brightness'] > 127:
            should_invert = True

        st.write("Auto-invert decided:", should_invert)

        # final 28x28 selection: if we inverted earlier, final is already white-on-black; else invert now
        final28 = steps['final28']
        final_arr = steps['final_arr_pre_invert']

        # If the 28x28 has a black background and black digit (digit==0), we need to invert
        # We'll compute mean of center region to guess.
        center_mean = np.array(final28)[8:20,8:20].mean()
        st.write("Center mean (0..255) on final28:", float(center_mean))

        if should_invert:
            final28 = ImageOps.invert(final28)
            final_arr = 1.0 - final_arr

        # show final
        st.write("Final 28x28 fed to model (zoomed):")
        st.image(final28.resize((140,140)), clamp=True)

        # prepare array properly shaped
        final_input = final_arr.reshape(1,28,28,1).astype(np.float32)

        # Predict
        probs = model.predict(final_input)[0]
        pred = np.argmax(probs)
        st.success(f"Predicted: {pred}  (confidence {probs[pred]*100:.2f}%)")

        st.write("Top 3:")
        for i in probs.argsort()[-3:][::-1]:
            st.write(f"{i}: {probs[i]*100:.2f}%")

# Save sample (only raw image saved)
if st.button("Save Raw Sample"):
    if canvas_result.image_data is None:
        st.warning("Draw something first")
    else:
        rgba = (canvas_result.image_data).astype("uint8")
        gray = composite_canvas_rgba_to_gray(rgba)
        ts = int(time.time()*1000)
        fname = os.path.join(SAVE_DIR, f"raw_{ts}.png")
        Image.fromarray(gray).save(fname)
        st.success(f"Saved raw grayscale sample as {fname}")

st.caption("Debug mode: shows raw → inverted → threshold → bbox → final28. Use auto/manual invert and threshold slider.")
