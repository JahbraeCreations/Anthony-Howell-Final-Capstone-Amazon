"""
Capstone Web Application
========================
Integrates all 5 models into a single web interface using Streamlit.

Run locally:  streamlit run webapp/app.py
Deploy:       Push to GitHub, then connect to Streamlit Community Cloud
              https://streamlit.io/cloud (free hosting)
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Page config
st.set_page_config(
    page_title="Aegis Health Strategy",
    page_icon="⚕️",
    layout="wide",
)

st.title("Aegis Health Strategy")
st.write("")

# Sidebar navigation
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Home",
        "Model 1: Traditional ML",
        "Model 2: Deep Learning",
        "Model 3: CNN (Image Classification)",
        "Model 4: NLP (Text Classification)",
        "Model 5: XGBoost",
        "Our Team" 
    ],
)

# ---------------------------------------------------------------------------
# Helper: Cache model loading so it only happens once
# ---------------------------------------------------------------------------
# Use @st.cache_resource for models — they load once and stay in memory.
#
# Example:
#     @st.cache_resource
#     def load_model1():
#         import joblib
#         return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
#
#     @st.cache_resource
#     def load_model3():
#         import tensorflow as tf
#         return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")

# ---------------------------------------------------------------------------
# Model pages — fill these in with your model loading and prediction logic
# ---------------------------------------------------------------------------

if model_choice == "Home":
    st.write("Welcome! Use the sidebar to navigate between models.")
    st.markdown("""
## Transforming Healthcare Through Intelligent Analytics

At **Aegis Health Strategy**, we specialize in leveraging artificial intelligence, machine learning, and advanced healthcare analytics to help medical organizations make smarter, faster, and more impactful clinical decisions.

Our mission is to empower healthcare providers with data-driven solutions that improve patient outcomes, optimize operational efficiency, and reduce preventable healthcare costs.

Through innovative predictive modeling, clinical decision support systems, and AI-powered diagnostics, we help healthcare organizations tackle critical challenges such as:

- **Patient Readmission Risk Prediction**  
- **Diabetic Retinopathy Detection**  
- **Medication Effectiveness Analysis**  
- **Clinical Resource Optimization**  
- **Healthcare Data Intelligence**

By combining cutting-edge machine learning models with real-world clinical insight, Aegis Health Strategy bridges the gap between healthcare data and actionable patient care strategies.

### Our Commitment
We believe healthcare innovation should be:
- **Accurate** — delivering reliable predictive insights  
- **Transparent** — providing interpretable AI for clinical trust  
- **Efficient** — streamlining workflows and improving care delivery  
- **Scalable** — built for long-term healthcare transformation  

Explore our platform to discover how intelligent healthcare solutions can drive better decisions, better care, and better outcomes.

**Protecting Patients. Powering Decisions. Advancing Healthcare.**
""")

elif model_choice == "Model 1: Traditional ML":
    st.header("Model 1: Traditional ML")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    # @st.cache_resource
    # def load_model1():
    #     import joblib
    #     return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
    #
    # model = load_model1()
    #
    # # Create input fields for your features
    # col1, col2 = st.columns(2)
    # with col1:
    #     feature_1 = st.number_input("Feature 1", value=0.0)
    #     feature_2 = st.selectbox("Feature 2", ["Option A", "Option B"])
    # with col2:
    #     feature_3 = st.slider("Feature 3", 0, 100, 50)
    #
    # if st.button("Predict"):
    #     import pandas as pd
    #     input_df = pd.DataFrame([{"feature_1": feature_1, ...}])
    #     prediction = model.predict(input_df)
    #     probability = model.predict_proba(input_df)
    #     st.success(f"Prediction: {prediction[0]}")
    #     st.write(f"Confidence: {probability.max():.2%}")
    # ---- END PATTERN ----

    st.info("Not yet implemented — load your model and add input fields here.")

elif model_choice == "Model 2: Deep Learning":
    st.header("Model 2: Deep Learning")
    # TODO: Load your DNN model and add prediction interface
    # Same pattern as Model 1, but load with:
    #     import tensorflow as tf
    #     model = tf.keras.models.load_model("models/model2_deep_learning/saved_model/model.keras")
    st.info("Not yet implemented — load your model and add input fields here.")

elif model_choice == "Model 3: CNN (Image Classification)":
    st.header("Model 3: CNN — Image Classification")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    # @st.cache_resource
    # def load_model3():
    #     import tensorflow as tf
    #     return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")
    #
    # model = load_model3()
    #
    # uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    # if uploaded_file is not None:
    #     from PIL import Image
    #     import numpy as np
    #
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption="Uploaded Image", use_container_width=True)
    #
    #     # Preprocess — must match your training preprocessing
    #     img_resized = image.resize((224, 224))
    #     img_array = np.array(img_resized) / 255.0
    #     img_batch = np.expand_dims(img_array, axis=0)
    #
    #     if st.button("Classify"):
    #         prediction = model.predict(img_batch)
    #         confidence = float(prediction.max())
    #         predicted_class = "Positive" if prediction[0][0] > 0.5 else "Negative"
    #         st.success(f"Prediction: {predicted_class}")
    #         st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

    st.info("Not yet implemented — add image upload and classification here.")

elif model_choice == "Model 4: NLP (Text Classification)":
    st.header("Model 4: NLP — Text Classification")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    # @st.cache_resource
    # def load_model4():
    #     import joblib
    #     model = joblib.load("models/model4_nlp_classification/saved_model/model.joblib")
    #     vectorizer = joblib.load("models/model4_nlp_classification/saved_model/vectorizer.joblib")
    #     return model, vectorizer
    #
    # model, vectorizer = load_model4()
    #
    # user_text = st.text_area("Enter text to classify:", height=150)
    # if st.button("Classify") and user_text:
    #     text_vectorized = vectorizer.transform([user_text])
    #     prediction = model.predict(text_vectorized)[0]
    #     confidence = model.predict_proba(text_vectorized).max()
    #     st.success(f"Predicted Category: {prediction}")
    #     st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

    st.info("Not yet implemented — add text input and classification here.")

elif model_choice == "Model 5: XGBoost":
    st.header("Model 5: XGBoost Diabetes Medication Predictor")
    # TODO: Add your custom model interface
    st.info("This model predicts whether a patient is likely to need diabetes medication based on non-medication clinical and demographic information.")
    
    @st.cache_resource
    def load_model5():
        
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        model_path = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model" / "model.joblib"
        return joblib.load(model_path)

    model = load_model5()

    st.subheader("Enter Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_options = {
            "0–10": 5,
            "10–20": 15,
            "20–30": 25,
            "30–40": 35,
            "40–50": 45,
            "50–60": 55,
            "60–70": 65,
            "70–80": 75,
            "80–90": 85,
            "90–100": 95
        }

        age_label = st.selectbox(
            "Age Range",
            list(age_options.keys())
        )

        age = age_options[age_label]

        time_in_hospital = st.number_input("Time in Hospital", min_value=1, value=14)
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, value=132)
        #need to figure differences between number lab procedures and number of procedures
        num_procedures = st.number_input("Number of Procedures", min_value=0, value=6)

        diag_1 = st.selectbox(
            "Primary Diagnosis Category",
            [
                "circulatory",
                "diabetes",
                "digestive",
                "external",
                "missing",
                "other",
                "respiratory"
            ]
        )

        diag_2 = st.selectbox(
            "Secondary Diagnosis Category",
            [
                "circulatory",
                "diabetes",
                "digestive",
                "external",
                "missing",
                "other",
                "respiratory"
            ]
        )

        admission_type = st.selectbox(
            "Admission Type",
            [2, 3, 4, 5, 6, 7, 8]
        )
        

    with col2:
        num_medications = st.number_input("Number of Medications", min_value=1, value=81)
        number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, value=42)
        number_emergency = st.number_input("Number of Emergency Visits", min_value=0, value=76)
        number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, value=21)

        race = st.selectbox(
            "Race",
            [
                "AfricanAmerican",
                "Asian",
                "Caucasian",
                "Hispanic"
            ]
        )

        diag_3 = st.selectbox(
            "Third Diagnosis Category",
            [
                "circulatory",
                "diabetes",
                "digestive",
                "external",
                "missing",
                "other",
                "respiratory"
            ]
        )

        discharge_disposition = st.selectbox(
            "Discharge Disposition",
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 22, 23, 24, 25, 27, 28]
        )

    with col3:
        number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, value=16)
        
        max_glu_options = {
            "Not Tested": -1,
            "Normal": 0,
            "High (>200 mg/dL)": 1,
            "Very High (>300 mg/dL)": 2
        }

        max_glu_label = st.selectbox(
            "Maximum Glucose Serum Result",
            list(max_glu_options.keys())
        )

        max_glu_serum = max_glu_options[max_glu_label]

        a1c_options = {
            "Not Tested": -1,
            "Normal": 0,
            "Elevated (>7%)": 1,
            "High (>8%)": 2
        }

        a1c_label = st.selectbox(
            "A1C Test Result",
            list(a1c_options.keys())
        )

        A1Cresult = a1c_options[a1c_label]

        gender = st.selectbox("Gender", ["Male", "Female"])

        gender_encoded = 0 if gender == "Male" else 1

        weight_checked = st.selectbox(
            "Weight Recorded?",
            ["No", "Yes"]
        )

        weight_checked = 1 if weight_checked == "Yes" else 0

        readmission_binary = st.selectbox(
            "Readmitted Previously?",
            ["No", "Yes"]
        )

        readmission_binary = 1 if readmission_binary == "Yes" else 0

        admission_source = st.selectbox(
            "Admission Source",
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 20, 22, 25]
        )





    

    # Build input row
    input_df = pd.DataFrame([{
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "gender": gender_encoded,
        "weight_checked": weight_checked,
        "readmission_binary": readmission_binary,
    }])

    # Add missing columns expected by model
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    race_col = f"race_{race}"

    if race_col in input_df.columns:
        input_df[race_col] = 1

    diag_1_col = f"diag_1_{diag_1}"

    if diag_1_col in input_df.columns:
        input_df[diag_1_col] = 1
    
    diag_2_col = f"diag_2_{diag_2}"

    if diag_2_col in input_df.columns:
        input_df[diag_2_col] = 1

    diag_3_col = f"diag_3_{diag_3}"

    if diag_3_col in input_df.columns:
        input_df[diag_3_col] = 1

    admission_type_col = f"admission_type_id_{admission_type}"

    if admission_type_col in input_df.columns:
        input_df[admission_type_col] = 1

    discharge_col = f"discharge_disposition_id_{discharge_disposition}"

    if discharge_col in input_df.columns:
        input_df[discharge_col] = 1

    admission_source_col = f"admission_source_id_{admission_source}"

    if admission_source_col in input_df.columns:
        input_df[admission_source_col] = 1

    # Reorder columns to match training
    input_df = input_df[list(model.feature_names_in_)]

    if st.button("Predict Diabetes Medication Need"):
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(input_df)[:, 1][0]
        else:
            confidence = None

        if prediction == 1:
            st.success("Prediction: Patient is likely to need diabetes medication.")
        else:
            st.warning("Prediction: Patient is not likely to need diabetes medication.")

        if confidence is not None:
            st.write(f"Confidence: {confidence:.2%}")



elif model_choice == "Our Team":
    st.header("Our Team")
    st.write("Please meet the members of Aegis Health Strategy")
    st.info("Clifton Rand: Cliff is our data engineer lead as well as our Model 5 lead")
    st.info("Jesse Goff: Jesse is our CNN model lead as well as our Streamlit lead")
    st.info("Sean McManus: Sean is our Traditional ML lead as well as our Presentation lead")
    st.info("Brodie Ellis: Brodie is our NLP lead")