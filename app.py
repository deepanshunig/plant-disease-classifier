"""Local leaf-photo classifier interface."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.common import ROOT, class_names, readable_label
from src.predict import Predictor, decode_image

st.set_page_config(page_title="Leaf condition classifier", page_icon="🌿", layout="centered")
st.title("🌿 Leaf condition classifier")
st.write("Upload a clear photo of one leaf to explore its predicted plant and condition.")


@st.cache_resource
def load_predictor(path: str, modified: int):
    return Predictor(path)


bundle = ROOT / "artifacts/model_bundle/model.pt"
with st.expander("Supported plants and conditions"):
    rows = [{"Plant": readable_label(name)[0], "Condition": readable_label(name)[1]} for name in class_names()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if not bundle.is_file():
    st.info("The model is being prepared. Predictions will become available after training completes.")
    st.stop()

try:
    predictor = load_predictor(str(bundle), bundle.stat().st_mtime_ns)
except Exception:
    st.error("The model could not be loaded. Check the local model bundle before predicting.")
    st.stop()

if predictor.checkpoint["training_profile"] == "pilot":
    st.warning("This is a preliminary model trained on a small sample. Its predictions are for exploring the demo.")
st.caption("Predictions cover the listed classes. Model scores can be high for unsupported inputs.")
uploaded = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"], max_upload_size=10)
if uploaded is not None:
    try:
        image = decode_image(uploaded.getvalue())
    except ValueError as error:
        st.error(str(error))
        st.stop()
    st.image(image, caption="Your leaf photo", width="stretch")
    if st.button("Classify leaf", type="primary"):
        with st.spinner("Examining the leaf…"):
            result = predictor.predict(image)
        prediction = result["prediction"]
        st.subheader(f"{prediction['crop']} · {prediction['condition']}")
        st.metric("Top model score", f"{prediction['score']:.1%}")
        if result["uncertain"]:
            st.warning("The model is uncertain about this photo. Try a closer, clearer view of the leaf.")
        alternatives = [{"Plant": item["crop"], "Condition": item["condition"], "Model score": f"{item['score']:.1%}"} for item in result["top_three"]]
        st.dataframe(pd.DataFrame(alternatives), hide_index=True, width="stretch")
        st.caption("Scores describe the model's ranking of the supported classes.")
