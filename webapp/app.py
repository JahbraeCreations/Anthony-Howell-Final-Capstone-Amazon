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

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
  
    @st.cache_resource
    def load_model3():
        import tensorflow as tf

        model_url = "https://huggingface.co/jfranklingoff/Capstone-Project-CNN-Model/resolve/main/cnn_model.keras"

        local_model_path = tf.keras.utils.get_file(
            "cnn_model.keras",
            origin=model_url
        )

        return tf.keras.models.load_model(local_model_path)
    
    model = load_model3()
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        from PIL import Image
        import numpy as np
    
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
        # Preprocess — must match your training preprocessing
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
    
        if st.button("Classify"):
            prediction = model.predict(img_batch)
            confidence = float(prediction.max())
            predicted_class = "Positive" if prediction[0][0] > 0.5 else "Negative"
            st.success(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

    st.info("""
            Add a retinal scan image file by dragging and dropping a file, or select a file using the Browse files button.

            Once uploaded, your selected image will be displayed. Click Classify to predict a classification with confidence score.
            """
            )

elif model_choice == "Model 4: NLP (Text Classification)":
    st.header("Model 4: NLP — Text Classification")

    import re

    # Must match the cleaning used in models/model4_nlp_classification/predict.py
    CONTRACTIONS = [
        (r"won't", "will not"),
        (r"can't", "can not"),
        (r"shan't", "shall not"),
        (r"n't", " not"),
        (r"'re", " are"),
        (r"'ve", " have"),
        (r"'ll", " will"),
        (r"'d", " would"),
        (r"'m", " am"),
        (r"it's", "it is"),
        (r"that's", "that is"),
        (r"what's", "what is"),
        (r"there's", "there is"),
    ]

    def clean_text(text):
        if not isinstance(text, str) or text.strip() == '':
            return ''
        text = text.lower()
        for pat, repl in CONTRACTIONS:
            text = re.sub(pat, repl, text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'%u[0-9a-fA-F]{4}', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @st.cache_resource
    def load_model4():
        model_dir = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
        model = joblib.load(model_dir / "model.joblib")
        vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
        return model, vectorizer

    model, vectorizer = load_model4()

    user_text = st.text_area("Enter patient medication feedback to classify:", height=150)
    if st.button("Classify") and user_text:
        cleaned = clean_text(user_text)
        text_vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized).max()
        st.success(f"Predicted Effectiveness: {prediction}")
        st.write(f"Confidence: {confidence:.2%}")

elif model_choice == "Model 5: XGBoost":
    st.header("Model 5: XGBoost Diabetes Medication Predictor")
    # TODO: Add your custom model interface
    st.info("This model predicts whether a patient is likely to need diabetes medication based on non-medication clinical and demographic information.")
    
    @st.cache_resource
    def load_model5():
        
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        model_path = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model" / "model.joblib"
        return joblib.load(model_path)

    #load model 5
    model = load_model5()

    st.subheader("Enter Patient Information")

    #creates 3 columns for cleaner ui
    col1, col2, col3 = st.columns(3)

    #col 1
    with col1:

        # Convert age range selections into numeric midpoint values
        # Matches preprocessing logic used during training
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

        # User selects age range
        age_label = st.selectbox(
            "Age Range",
            list(age_options.keys())
        )

        # Convert selected age range into numeric value
        age = age_options[age_label]

        # Binary encode whether weight was recorded
        weight_checked = st.selectbox(
            "Weight Recorded?",
            ["No", "Yes"]
        )
        weight_checked = 1 if weight_checked == "Yes" else 0

        # Numeric hospital visit features
        number_emergency = st.number_input("Number of Emergency Visits", min_value=0, max_value=76, value=38)

        number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=16, value=8)

        #Primary diagnosis category selection
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

        #Admission source category
        admission_source = st.selectbox(
            "Admission Source",
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 20, 22, 25]
        )

        # Glucose result options mapped to ordinal encoding
        # Matches training pipeline encoding
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


    with col2:
        
        #Binary encode gender to match training data
        gender = st.selectbox("Gender", ["Male", "Female"])

        gender_encoded = 0 if gender == "Male" else 1

        # Binary encode previous readmission history
        readmission_binary = st.selectbox(
            "Readmitted Previously?",
            ["No", "Yes"]
        )

        readmission_binary = 1 if readmission_binary == "Yes" else 0

        # Numeric clinical features
        number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, max_value=21, value=11)
        
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, max_value=132, value=65)

        # Secondary diagnosis category
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

        # Admission type category
        admission_type = st.selectbox(
            "Admission Type",
            [2, 3, 4, 5, 6, 7, 8]
        )



    with col3:

        # Race category (used for one-hot encoding)
        race = st.selectbox(
            "Race",
            [
                "AfricanAmerican",
                "Asian",
                "Caucasian",
                "Hispanic"
            ]
        )

        # Hospital stay length
        time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=14, value=7)

        # Outpatient visit history
        number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, max_value=42, value=21)

        # Number of procedures performed
        num_procedures = st.number_input("Number of Medical Procedures", min_value=0, max_value=6, value=3)
        
        # Third diagnosis category
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

        # Discharge disposition category
        discharge_disposition = st.selectbox(
            "Discharge Disposition",
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 22, 23, 24, 25, 27, 28]
        )

        # A1C result mapped to ordinal encoding
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



    # Create single-row DataFrame from user inputs
    # This matches the model input format
    input_df = pd.DataFrame([{
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
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

    #ensures non selected items in select boxes are set to 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # One-hot encode selected race value
    race_col = f"race_{race}"

    if race_col in input_df.columns:
        input_df[race_col] = 1

    # One-hot encode diagnosis categories
    diag_1_col = f"diag_1_{diag_1}"

    if diag_1_col in input_df.columns:
        input_df[diag_1_col] = 1
    
    diag_2_col = f"diag_2_{diag_2}"

    if diag_2_col in input_df.columns:
        input_df[diag_2_col] = 1

    diag_3_col = f"diag_3_{diag_3}"

    if diag_3_col in input_df.columns:
        input_df[diag_3_col] = 1

    # One-hot encode admission type
    admission_type_col = f"admission_type_id_{admission_type}"

    if admission_type_col in input_df.columns:
        input_df[admission_type_col] = 1

    # One-hot encode discharge disposition
    discharge_col = f"discharge_disposition_id_{discharge_disposition}"

    if discharge_col in input_df.columns:
        input_df[discharge_col] = 1

    # One-hot encode admission source
    admission_source_col = f"admission_source_id_{admission_source}"

    if admission_source_col in input_df.columns:
        input_df[admission_source_col] = 1

    # Reorder columns to match expected input order
    input_df = input_df[list(model.feature_names_in_)]

    # Run prediction when button is clicked
    if st.button("Predict Diabetes Medication Need"):
        
        # Get probability for positive class (needs medication)
        confidence = model.predict_proba(input_df)[:, 1][0]

        # Use tuned threshold from model evaluation
        # Default XGBoost threshold is 0.50, but 0.70 gave better class balance
        threshold = 0.70

        # Convert probability into final prediction
        prediction =1 if confidence >= threshold else 0

        # Display prediction result to user
        if prediction == 1:
            st.success("Prediction: Patient is likely to need diabetes medication.")
        else:
            st.warning("Prediction: Patient is not likely to need diabetes medication.")

        # Display model confidence as percentage
        st.write(f"confidence: {confidence:.2%}")



elif model_choice == "Our Team":
    st.header("Our Team")
    st.write("Please meet the members of Aegis Health Strategy")
    st.info("Clifton Rand: Cliff is our Data Engineer and Model 5 lead")
    st.info("Jesse Goff: Jesse is our Model 3 and Presentation lead")
    st.info("Sean McManus: Sean is our Model 1 lead")
    st.info("Brodie Ellis: Brodie is our Model 4 lead")