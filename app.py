import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import time
from preprocessing import preprocess_data
from config import MODEL_PATH, CATEGORY_ENCODER_PATH, GENDER_ENCODER_PATH

# --- Configuration ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# --- Load Model and Encoders ---
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_encoder(encoder_path):
    try:
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        return encoder
    except FileNotFoundError:
        st.error(f"Error: Encoder file not found at {encoder_path}")
        return None
    except Exception as e:
        st.error(f"Error loading encoder: {e}")
        return None

model = load_model(MODEL_PATH)
category_encoder = load_encoder(CATEGORY_ENCODER_PATH)
gender_encoder = load_encoder(GENDER_ENCODER_PATH)

# Check if model and encoders loaded successfully
if model is None or category_encoder is None or gender_encoder is None:
    st.stop()

# --- Streamlit UI ---
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application predicts the probability of a credit card transaction being fraudulent
based on a pre-trained LightGBM model. You can predict for a single transaction or
upload a small batch file (CSV).
""")

# --- Input Mode Selection ---
input_mode = st.sidebar.radio("Select Input Mode:", ("Single Transaction", "Batch Upload (CSV)"))

# --- Prediction Logic ---
final_feature_df = None
original_input_df = None
predictions = None
probabilities = None

if input_mode == "Single Transaction":
    st.sidebar.header("Enter Transaction Details:")
    with st.sidebar.form(key='single_txn_form'):
        cc_num = st.text_input("Credit Card Number", "1234567890123456")
        amt = st.number_input("Amount", min_value=0.01, value=50.0, step=1.0, format="%.2f")
        known_categories = list(category_encoder.classes_)
        category = st.selectbox("Category", options=known_categories, index=0)
        known_genders = list(gender_encoder.classes_)
        gender = st.selectbox("Gender", options=known_genders, index=0)
        lat = st.number_input("Cardholder Latitude", value=40.7128, format="%.4f")
        long = st.number_input("Cardholder Longitude", value=-74.0060, format="%.4f")
        city_pop = st.number_input("Cardholder City Population", min_value=1, value=8000000, step=1000)
        unix_time = st.number_input("Transaction Unix Timestamp", value=int(time.time()), step=1)
        merch_lat = st.number_input("Merchant Latitude", value=34.0522, format="%.4f")
        merch_long = st.number_input("Merchant Longitude", value=-118.2437, format="%.4f")
        submit_button = st.form_submit_button(label='Predict Fraud')

    if submit_button:
        input_data = {
            'cc_num': [cc_num],
            'amt': [amt],
            'category': [category],
            'gender': [gender],
            'lat': [lat],
            'long': [long],
            'city_pop': [city_pop],
            'unix_time': [unix_time],
            'merch_lat': [merch_lat],
            'merch_long': [merch_long]
        }
        original_input_df = pd.DataFrame(input_data)
        st.subheader("Processing Input:")
        st.dataframe(original_input_df)

        try:
            with st.spinner("Preprocessing data..."):
                final_feature_df, processed_df_full = preprocess_data(original_input_df, category_encoder, gender_encoder)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        if final_feature_df is not None:
            try:
                predictions = model.predict(final_feature_df)
                probabilities = model.predict_proba(final_feature_df)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.stop()

elif input_mode == "Batch Upload (CSV)":
    st.sidebar.header("Upload Transaction File:")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    max_rows = st.sidebar.slider("Max rows to process:", min_value=1, max_value=50, value=10)

    if uploaded_file is not None:
        try:
            original_input_df = pd.read_csv(uploaded_file, nrows=max_rows)
            st.subheader(f"Processing first {min(max_rows, len(original_input_df))} rows of uploaded data:")
            st.dataframe(original_input_df.head(max_rows))

            try:
                with st.spinner(f"Preprocessing {len(original_input_df)} rows..."):
                    final_feature_df, processed_df_full = preprocess_data(original_input_df, category_encoder, gender_encoder)
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                st.stop()

            if final_feature_df is not None:
                try:
                    predictions = model.predict(final_feature_df)
                    probabilities = model.predict_proba(final_feature_df)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.stop()

        except Exception as e:
            st.error(f"Error reading or processing CSV file: {e}")

# --- Display Results ---
st.subheader("Prediction Results")

if predictions is not None and probabilities is not None and final_feature_df is not None:
    display_df = original_input_df.loc[final_feature_df.index].copy() if original_input_df is not None else pd.DataFrame()
    display_df['Prediction'] = ['Fraud' if p == 1 else 'Not Fraud' for p in predictions]
    display_df['Fraud Probability'] = probabilities[:, 1]
    st.dataframe(display_df)

    if input_mode == "Batch Upload (CSV)" and len(display_df) > 1:
        st.markdown("---")
        st.subheader("Batch Summary")
        col1, col2 = st.columns(2)
        with col1:
            fraud_counts = display_df['Prediction'].value_counts()
            st.metric("Total Transactions Processed", len(display_df))
            st.metric("Predicted Fraudulent", fraud_counts.get('Fraud', 0))
            
        with col2:
            st.markdown("**Prediction Distribution**")
            if not fraud_counts.empty:
                st.bar_chart(fraud_counts)
            else:
                st.write("No predictions to display.")

    elif input_mode == "Single Transaction" and len(display_df) == 1:
        pred_label = display_df['Prediction'].iloc[0]
        pred_prob = display_df['Fraud Probability'].iloc[0]
        if pred_label == "Fraud":
            st.error(f"Prediction: {pred_label} (Probability: {pred_prob:.2%})")
        else:
            st.success(f"Prediction: {pred_label} (Probability: {pred_prob:.2%})")
