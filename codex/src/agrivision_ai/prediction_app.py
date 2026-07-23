from pathlib import Path
import tempfile
import time

import streamlit as st
from PIL import Image

from prediction_backend import AgriVisionPredictor
from plant_tips import get_tip

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================

MODEL_PATH = Path("codex/models/best_model.keras")
DATASET_CONFIG = Path("codex/config/dataset_config.json")
DISEASE_DATABASE = Path("codex/config/disease_database.json")

# =====================================================
# LOAD PREDICTOR
# =====================================================

@st.cache_resource
def load_predictor():

    return AgriVisionPredictor(
        model_path=MODEL_PATH,
        dataset_config_path=DATASET_CONFIG,
        disease_database_path=DISEASE_DATABASE,
    )

# =====================================================
# CSS
# =====================================================

def load_css():

    st.markdown("""
    <style>

    .stApp{
        background:#F4F8F3;
    }

    section[data-testid="stSidebar"]{
        background:#EAF5E7;
    }

    h1{
        color:#2E7D32;
        font-weight:700;
    }

    .main-title{
        text-align:center;
        font-size:55px;
        color:#2E7D32;
        font-weight:800;
    }

    .sub-title{
        text-align:center;
        color:#666666;
        font-size:22px;
        margin-bottom:30px;
    }

    </style>
    """, unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

def render_sidebar():

    with st.sidebar:

        st.title("🌿 AgriVision AI")

        st.success("AI Model Loaded")

        st.divider()

        st.subheader("Features")

        st.write("• 44 Plant Diseases")
        st.write("• Healthy Plant Detection")
        st.write("• Confidence Score")
        st.write("• Treatment Suggestions")
        st.write("• Prevention Tips")



        st.divider()

        st.caption("Developer")

        st.write("**Dhruvil Jain**")

        st.caption("Science Vasudha (& NCSC) Project")

        # =====================================================
# HEADER
# =====================================================

def render_header():

    st.markdown(
        "<div class='main-title'>🌿 AgriVision AI</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sub-title'>AI Powered Plant Disease Detection</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "Upload a clear image of a single plant leaf. "
        "The AI model will identify the plant, detect diseases, "
        "estimate confidence, and display treatment information."
    )


# =====================================================
# UPLOAD SECTION
# =====================================================

def upload_leaf():

    st.subheader("📷 Upload Plant Leaf")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return None, uploaded_file

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1])

    with left:

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    with right:

        st.success("Image Loaded Successfully")

        st.write("Filename:")

        st.code(uploaded_file.name)

        st.write("Press the button below to start analysis.")

        analyze = st.button(
            "🔍 Analyze Leaf",
            use_container_width=True
        )

    if not analyze:
        return None, None

    return image, uploaded_file


# =====================================================
# RUN PREDICTION
# =====================================================

def predict_image(predictor, image):

    with st.spinner("Analyzing leaf..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            image.save(tmp.name)

            result = predictor.predict(Path(tmp.name))

    return result

# =====================================================
# RESULT SECTION
# =====================================================

def show_result(result, prediction_time):

    st.divider()

    st.subheader("🌱 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌿 Plant",
            result.plant_species.title()
        )

    with col2:
        st.metric(
            "🦠 Disease",
            result.disease_name.title()
        )

    with col3:
        st.metric(
            "🎯 Confidence",
            f"{result.confidence_percentage:.2f}%"
        )

    with col4:
        st.metric(
            "⚡ Time",
            f"{prediction_time:.2f} sec"
        )
    st.markdown("### 🎯 Prediction Confidence")

    st.progress(result.confidence_percentage / 100)

    st.caption(f"{result.confidence_percentage:.2f}% Confidence")

    st.divider()
    if result.disease_information is None:

            info = {
        "short_description": "Detailed information is not available yet.",
        "cause": ["Not available"],
        "symptoms": ["Not available"],
        "treatment": ["Not available"],
        "prevention": ["Not available"]
    }
            
    else:

        from dataclasses import asdict

        info = result.disease_information

        if info is not None and hasattr(info, "__dataclass_fields__"):
            info = asdict(info)

    left, right = st.columns(2)

    with left:

        with st.expander("📝 Description", expanded=True):

            st.write(
                info.get(
                    "short_description",
                    "No description available."
                )
            )

        with st.expander("🦠 Cause"):

            for item in info.get("cause", []):

                st.write("•", item)

        with st.expander("⚠ Symptoms"):

            for item in info.get("symptoms", []):

                st.write("•", item)

    with right:

        with st.expander("💊 Treatment", expanded=True):

            for item in info.get("treatment", []):

                st.write("•", item)

        with st.expander("🛡 Prevention"):

            for item in info.get("prevention", []):

                st.write("•", item)

        st.divider()

        st.subheader("🌱 Plant Care Tip")

        st.info(get_tip(result.plant_species))

                # =====================================================
# FOOTER
# =====================================================

def render_footer():

    st.divider()

    st.caption(
        "🌿 AgriVision AI | "
        "Made using Python • TensorFlow • Streamlit"
    )

    st.caption(
        "Developed by Dhruvil Jain | "
        "Science Vasudha (& NCSC) Project"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    load_css()

    predictor = load_predictor()

    render_sidebar()

    render_header()

    image, uploaded_file = upload_leaf()

    if image is None:

        render_footer()

        return

    start_time = time.perf_counter()

    result = predict_image(
    predictor,
    image
)

    prediction_time = time.perf_counter() - start_time

    show_result(result, prediction_time)

    render_footer()


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    main()
