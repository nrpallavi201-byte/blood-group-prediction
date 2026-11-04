
import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2

from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model

from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as PDFImage, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as PDFImage, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Fingerprint Classification",
    page_icon="🔬",
    layout="wide"
)

# PROJECT PATHS
PROJECT_DIR = "/content/drive/MyDrive/Blood_Group_Prediction"

MODEL_PATH = os.path.join(
    PROJECT_DIR, "final_cnn_model.keras"
)

HISTORY_DIR = "/tmp/prediction_history"
)

HISTORY_FILE = os.path.join(
    HISTORY_DIR, "prediction_history.csv"
)

os.makedirs(HISTORY_DIR, exist_ok=True)

classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']


# PROFESSIONAL BLUE AND WHITE DESIGN

st.markdown('''
<style>
.stApp {
    background-color: #f3f7fc;
}
.main-title {
    text-align: center;
    color: #123b69;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
}
[data-testid="stSidebar"] {
    background-color: #e8f1fb;
}
.stButton > button, .stDownloadButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 9px;
    font-weight: bold;
}
[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #dce7f3;
}
</style>
''', unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">AI-Based Fingerprint Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Experimental CNN Image Classification System</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Academic research prototype only. Fingerprint images cannot "
    "medically determine actual blood group. Laboratory testing is required."
)

# LOAD MODEL FROM HUGGING FACE
@st.cache_resource
def get_model():
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="Pallavi-201/blood-group-cnn-model",
        filename="final_cnn_model.keras"
    )

    return load_model(model_path)

# IMAGE QUALITY CHECK
def check_image_quality(image):

    img = np.array(image.convert("L"))

    height, width = img.shape

    brightness = np.mean(img)

    sharpness = cv2.Laplacian(
        img, cv2.CV_64F
    ).var()

    brightness_status = (
        "Too Dark" if brightness < 40 else
        "Too Bright" if brightness > 220 else
        "Good"
    )

    clarity_status = (
        "Blurry" if sharpness < 50 else "Clear"
    )

    resolution_status = (
        "Low Resolution"
        if width < 96 or height < 96
        else "Good"
    )

    return brightness_status, clarity_status, resolution_status




# PDF REPORT GENERATION

def generate_pdf_report(image, image_name, label, score, prediction):

    from xml.sax.saxutils import escape
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=22,
        bottomMargin=22
    )

    styles = getSampleStyleSheet()
    content = []

    title_style = ParagraphStyle(
        "CompactTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#123B69"),
        fontSize=18,
        leading=21,
        alignment=TA_CENTER,
        spaceAfter=4
    )

    heading_style = ParagraphStyle(
        "CompactHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1769AA"),
        fontSize=11,
        leading=14,
        spaceBefore=5,
        spaceAfter=4
    )

    content.append(Paragraph(
        "AI-BASED FINGERPRINT CLASSIFICATION",
        title_style
    ))

    content.append(Paragraph(
        "Experimental CNN Image Classification Report",
        ParagraphStyle(
            "CompactSubtitle",
            parent=styles["Normal"],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
    ))

    content.append(Spacer(1, 6))

    content.append(Paragraph("1. Prediction Details", heading_style))

    details = [
        ["Report Date", datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        ["Image Name", escape(str(image_name))],
        ["Predicted Dataset Label", str(label)],
        ["Model Output Score", f"{score:.2f}%"]
    ]

    details_table = Table(details, colWidths=[180, 350])

    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#DCEBFA")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#123B69")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    content.append(details_table)

    content.append(Paragraph("2. Image Quality Report", heading_style))

    brightness, clarity, resolution = check_image_quality(image)

    quality_data = [
        ["Parameter", "Result"],
        ["Brightness", brightness],
        ["Clarity", clarity],
        ["Resolution", resolution]
    ]

    quality_table = Table(quality_data, colWidths=[265, 265])

    quality_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769AA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 4)
    ]))

    content.append(quality_table)

    content.append(Paragraph(
        "3. Uploaded Fingerprint Image",
        heading_style
    ))

    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    image_buffer.seek(0)

    content.append(PDFImage(
        image_buffer,
        width=1.35 * inch,
        height=1.35 * inch
    ))

    content.append(Paragraph(
        "4. Class-wise Model Output Scores",
        heading_style
    ))

    table_data = [["Dataset Label", "Output Score (%)"]]

    for class_name, class_score in zip(classes, prediction):
        table_data.append([
            class_name,
            f"{float(class_score) * 100:.2f}%"
        ])

    score_table = Table(table_data, colWidths=[265, 265])

    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769AA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 3)
    ]))

    content.append(score_table)

    content.append(Paragraph(
        "Important Scientific Disclaimer",
        heading_style
    ))

    disclaimer = (
        "This is an experimental academic image-classification prototype. "
        "Fingerprint images cannot medically determine or verify a person's "
        "actual blood group. Displayed scores are experimental model outputs, "
        "not medically verified probabilities. Laboratory testing is required "
        "for actual blood-group identification."
    )

    content.append(Paragraph(
        disclaimer,
        ParagraphStyle(
            "CompactDisclaimer",
            parent=styles["Normal"],
            fontSize=8,
            leading=11
        )
    ))

    pdf.build(content)

    buffer.seek(0)
    return buffer.getvalue()



# NAVIGATION
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Dashboard", "Prediction History"]
)

# HOME PAGE
if page == "Home":

    left, right = st.columns(2)

    with left:

        st.subheader("📤 Upload Fingerprint")

        uploaded_file = st.file_uploader(
            "Choose a fingerprint image",
            type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:

            original_image = Image.open(uploaded_file)

            st.image(
                original_image,
                caption="Uploaded Fingerprint",
                use_container_width=True
            )

            brightness, clarity, resolution = check_image_quality(
                original_image
            )

            st.subheader("🖼️ Image Quality Report")

            st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 23px;
                white-space: normal;
                overflow-wrap: anywhere;
            }
            </style>
            """, unsafe_allow_html=True)

            quality_col1, quality_col2, quality_col3 = st.columns(
                [1.1, 1.0, 1.3]
            )

            with quality_col1:
                st.metric("☀️ Brightness", brightness)

            with quality_col2:
                st.metric("🔍 Clarity", clarity)

            with quality_col3:
                st.metric("🖼️ Resolution", resolution)

            if brightness == "Good" and clarity == "Clear" and resolution == "Good":
                st.success("Image quality checks completed. All three checks passed.")
            else:
                st.warning("Some image quality checks need attention. Consider uploading a clearer image.")

    with right:

        st.subheader("🔍 Classification")

        if uploaded_file is not None:

            if st.button(
                "Analyze Fingerprint",
                use_container_width=True
            ):

                with st.spinner("Analyzing image..."):

                    model = get_model()

                    image = Image.open(
                        uploaded_file
                    ).convert("L")

                    image = image.resize((128, 128))

                    image_array = np.array(image) / 255.0

                    image_array = image_array.reshape(
                        1, 128, 128, 1
                    )

                    prediction = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    index = np.argmax(prediction)

                    label = classes[index]

                    score = prediction[index] * 100

                    # SAVE PREDICTION HISTORY
                    record = pd.DataFrame([{
                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Image Name": uploaded_file.name,
                        "Predicted Label": label,
                        "Model Output Score": round(
                            float(score), 2
                        )
                    }])

                    record.to_csv(
                        HISTORY_FILE,
                        mode="a",
                        header=not os.path.exists(HISTORY_FILE)
                        or os.path.getsize(HISTORY_FILE) == 0,
                        index=False
                    )

                st.success("Analysis completed! Prediction saved.")


                st.success("Analysis completed! Prediction saved.")

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #e4f1ff, #ffffff);
                        padding: 28px;
                        border-radius: 16px;
                        text-align: center;
                        border: 2px solid #b8d8f5;
                        margin-top: 15px;
                    ">
                        <h3 style="color:#174a7c;">
                            Experimental Classification Result
                        </h3>
                        <h1 style="
                            color:#0875c9;
                            font-size:55px;
                            margin:10px;
                        ">{label}</h1>
                        <h3 style="color:#24364b;">
                            Model Output Score: {score:.2f}%
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("📊 Detailed Class-wise Output Scores")

                for class_name, class_score in zip(classes, prediction):
                    percentage = float(class_score) * 100

                    st.write(
                        f"**{class_name}** — {percentage:.2f}%"
                    )

                    st.progress(float(class_score))

                st.warning(
                    "This is an experimental model output, "
                    "not a medically valid blood-group test. "
                    "Actual blood group requires laboratory testing."
                )

                pdf_report = generate_pdf_report(
                    original_image,
                    uploaded_file.name,
                    label,
                    score,
                    prediction
                )

                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_report,
                    file_name="Fingerprint_Classification_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


# DASHBOARD PAGE
elif page == "Dashboard":

    st.subheader("📊 Prediction Dashboard")

    st.markdown(
        "Overview of experimental fingerprint image classification records."
    )

    if os.path.exists(HISTORY_FILE):

        history_df = pd.read_csv(HISTORY_FILE)

        if not history_df.empty:

            # STATISTICS
            total = len(history_df)
            average = history_df["Model Output Score"].mean()
            common_label = history_df["Predicted Label"].mode()[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "📁 Total Predictions",
                total
            )

            col2.metric(
                "📊 Average Model Output Score",
                f"{average:.2f}%"
            )

            col3.metric(
                "🏷️ Most Common Dataset Label",
                common_label
            )

            st.divider()

            # LABEL DISTRIBUTION
            st.subheader("📈 Dataset Label Distribution")

            label_counts = (
                history_df["Predicted Label"]
                .value_counts()
                .reindex(classes, fill_value=0)
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**Prediction Count by Label**")
                st.bar_chart(label_counts)

            with chart_col2:
                st.markdown("**Prediction Percentage by Label**")
                st.bar_chart(
                    (label_counts / total * 100).rename(
                        "Percentage (%)"
                    )
                )

            st.divider()

            # RECENT PREDICTIONS
            st.subheader("🕒 Recent Five Predictions")

            recent_df = history_df.tail(5).iloc[::-1]

            st.dataframe(
                recent_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Note: These statistics summarize model outputs. "
                "They do not represent medically confirmed blood groups."
            )

        else:
            st.info("No prediction records available yet.")

    else:
        st.info("Prediction history is not available.")


# PREDICTION HISTORY PAGE
elif page == "Prediction History":

    st.subheader("📋 Prediction History")

    st.write("Search, filter and download your saved prediction records.")

    if os.path.exists(HISTORY_FILE):

        history_df = pd.read_csv(HISTORY_FILE)

        if not history_df.empty:

            st.metric(
                "Total Saved Predictions",
                len(history_df)
            )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                search_text = st.text_input(
                    "🔍 Search by image name",
                    placeholder="Enter image name..."
                )

            with col2:
                selected_label = st.selectbox(
                    "Filter by Dataset Label",
                    ["All Labels"] + classes
                )

            filtered_df = history_df.copy()

            if search_text:
                filtered_df = filtered_df[
                    filtered_df["Image Name"].str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                ]

            if selected_label != "All Labels":
                filtered_df = filtered_df[
                    filtered_df["Predicted Label"] == selected_label
                ]

            st.write(
                f"Showing {len(filtered_df)} of {len(history_df)} records"
            )

            st.dataframe(
                filtered_df,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "📥 Download Prediction History (CSV)",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="Prediction_History.csv",
                mime="text/csv",
                width="stretch"
            )

        else:
            st.info("No predictions saved yet.")

    else:
        st.info("Prediction history is not available.")

# FOOTER
st.divider()

st.markdown(
    "<center>Academic Project | CNN-Based Fingerprint Image Classification</center>",
    unsafe_allow_html=True
)
