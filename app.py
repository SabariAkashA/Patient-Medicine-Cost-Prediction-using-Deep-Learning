
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
import plotly.express as px

# ====== SETUP ======
st.set_page_config(
    page_title="Patient Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# ====== CUSTOM STYLES ======
st.markdown("""
<style>
    .input-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .input-title {
        color: #2c3e50;
        font-size: 1.2rem;
        margin-bottom: 15px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
    }
    .required-field::after {
        content: " *";
        color: red;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ====== MODEL LOADING ======
@st.cache_resource
def load_model_components():
    metrics.MeanSquaredError(name='mse')
    model = load_model('patient_cost_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mse')
    features = joblib.load('selected_features.pkl')
    return model, features

model, features = load_model_components()

# ====== MAIN APP ======
st.title("🏥 Patient Medical Cost Prediction")
st.markdown("Please fill in all required patient information to get an accurate cost estimate.")

# ====== PATIENT INFORMATION SECTION ======
with st.form("patient_info_form"):
    st.markdown('<div class="input-title">Patient Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="required-field">Basic Details</p>', unsafe_allow_html=True)
        age = st.slider("Age (years)", 0, 120, 30)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)

    with col2:
        st.markdown('<p class="required-field">Medical Identification</p>', unsafe_allow_html=True)
        blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
        insurance_provider = st.selectbox("Insurance Provider", ["Private", "Medicare", "Medicaid", "None"])

    # ====== MEDICAL HISTORY SECTION ======
    st.markdown('<div class="input-title">Medical History</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="required-field">Current Condition</p>', unsafe_allow_html=True)
        medical_condition = st.selectbox("Primary Diagnosis",
                                       ["Cancer", "Diabetes", "Heart Disease", "Respiratory", "Other"])
        symptoms = st.multiselect("Symptoms",
                                 ["Fever", "Pain", "Fatigue", "Nausea", "Swelling", "Other"])

    with col4:
        st.markdown('<p class="required-field">Treatment Plan</p>', unsafe_allow_html=True)
        admission_type = st.selectbox("Admission Type",
                                    ["Emergency", "Urgent", "Elective"])
        treatment_plan = st.selectbox("Proposed Treatment",
                                    ["Surgery", "Medication", "Therapy", "Observation"])

    # ====== HOSPITAL STAY DETAILS ======
    st.markdown('<div class="input-title">Hospitalization Details</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        planned_stay = st.slider("Planned Stay Duration (days)", 1, 30, 3)
        room_type = st.selectbox("Room Type",
                               ["General Ward", "Semi-Private", "Private", "ICU"])

    with col6:
        procedures = st.multiselect("Planned Procedures",
                                  ["MRI Scan", "X-Ray", "Blood Test", "Surgery", "Physical Therapy"])
        medications = st.multiselect("Required Medications",
                                   ["Antibiotics", "Painkillers", "Insulin", "Chemotherapy", "Other"])

    # ====== SUBMIT BUTTON ======
    submitted = st.form_submit_button("Calculate Estimated Cost", type="primary")

# ====== PREDICTION LOGIC ======
if submitted:
    with st.spinner('Processing your request...'):
        try:
            # Prepare input data mapping
            risk_map = {
                "Cancer": 3,
                "Diabetes": 2,
                "Heart Disease": 2,
                "Respiratory": 1,
                "Other": 1
            }

            input_data = {
                'age': [age],
                'gender': [1 if gender == "Male" else 0],
                'stay_length': [planned_stay],
                'risk_score': [risk_map[medical_condition]],
                f'blood type_{blood_type}': [1],
                f'medical condition_{medical_condition}': [1],
                f'admission type_{admission_type}': [1],
                'height': [height],
                'weight': [weight],
                f'insurance provider_{insurance_provider}': [1]
            }

            # Add procedure and medication flags
            for proc in ["MRI Scan", "X-Ray", "Blood Test", "Surgery", "Physical Therapy"]:
                input_data[f'procedure_{proc}'] = [1 if proc in procedures else 0]

            for med in ["Antibiotics", "Painkillers", "Insulin", "Chemotherapy", "Other"]:
                input_data[f'medication_{med}'] = [1 if med in medications else 0]

            # Create feature DataFrame
            df = pd.DataFrame(0, index=[0], columns=features)
            for col in input_data:
                if col in features:
                    df[col] = input_data[col]

            # Make prediction
            prediction = model.predict(df)[0][0]

            # Display results
            st.success("## Estimated Treatment Cost: ₹{:,.2f}".format(prediction))

            # Show input summary
            with st.expander("View Patient Input Summary"):
                st.write(f"**Age:** {age} years")
                st.write(f"**Gender:** {gender}")
                st.write(f"**Height/Weight:** {height}cm / {weight}kg")
                st.write(f"**Blood Type:** {blood_type}")
                st.write(f"**Insurance:** {insurance_provider}")
                st.write(f"**Condition:** {medical_condition} (Risk Score: {risk_map[medical_condition]})")
                st.write(f"**Admission Type:** {admission_type}")
                st.write(f"**Planned Stay:** {planned_stay} days in {room_type}")
                st.write(f"**Procedures:** {', '.join(procedures) if procedures else 'None'}")
                st.write(f"**Medications:** {', '.join(medications) if medications else 'None'}")

            # Cost breakdown visualization
            st.subheader("Cost Breakdown")
            cost_data = {
                "Component": ["Base Hospitalization", "Medical Condition",
                             "Procedures", "Medications", "Room Type"],
                "Amount": [
                    prediction * 0.4,
                    prediction * 0.3,
                    prediction * 0.15,
                    prediction * 0.1,
                    prediction * 0.05
                ]
            }
            fig = px.pie(cost_data, values="Amount", names="Component",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check if all required fields are filled correctly")