import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# Show the page title and description.
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁")
st.title("🫁 Lung Cancer Detection")
st.write(
    """
    Lung cancer is a significant contributor to cancer-related mortality. 
    With recent advancements in Computer Vision, Vision Transformers have gained traction and shown remarkable success in medical image analysis. 
    This study explores the potential of Vision Transformer models (ViT, CvT, CCT ViT, Parallel-ViT, Efficient ViT) compared to established state-of-the-art 
    architectures (CNN) for lung cancer detection 
    via medical imaging modalities, including CT scans and X-rays. This work will evaluate the impact of data availability and different training approaches 
    on model performance. The training approaches considered include but are not limited to Supervised Learning and Transfer Learning. 
    Established evaluation metrics such as accuracy, recall, precision, F1-score, and area under the ROC curve (AUC-ROC) will assess model performance in 
    detection efficacy, data validity, and computational efficiency. Cost-sensitive evaluation metrics such as cost matrix and weighted loss will analyse model performance 
    by considering the real-world implications of different types of errors, especially in cases where misdiagnosing a cancer case is more critical.
    """ 
)

# --------------------------------------------------------------

st.subheader("CT Scans of Lung Cancer")
st.image("images/Lung Cancer Images/CT/CT.png", caption="Sample CT Scan Images Used for Model Training in Lung Cancer Detection")

st.subheader("Histopathological Images of Lung Cancer")
st.image("images/Lung Cancer Images/Histopathological/Histopathological.png", caption="Sample  Images Used for Model Training in Lung Cancer Detection")

# --------------------------------------------------------------

st.subheader("🔍 Exploring Lung Cancer")
st.write(
    """
    This section visualises data from the [Exploring Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer/data).
    The effectiveness of cancer prediction system can inform individuals of their cancer risk with low cost and it will help people to make a more informed decision based on their cancer risk status. 
    Just click on the widgets below to explore!
    """
)

cancer_directory = "data/survey_lung_cancer.csv"
@st.cache_data
def load_data():
    lung_df = pd.read_csv(cancer_directory)
    lung_df.columns = lung_df.columns.str.replace('_', ' ').str.strip().str.title()
    return lung_df

lung_df = load_data()


print(lung_df.columns)

# Mapping dictionary for binary columns (1: No, 2: Yes)
binary_mapping = {1: "No", 2: "Yes", "YES": "Yes", "NO": "No", "M": "Male", "F": "Female"}

columns = ["Gender", "Age"]

# Apply mapping to each binary column
for col in lung_df.columns:
    if col != "Age":
        lung_df[col] = lung_df[col].map(binary_mapping)

# Gender selection with mapped values
genders = st.multiselect(
    "**Select Gender**",
    options=lung_df["Gender"].unique().tolist(),
    default=["Male", "Female"]
)

main_features = ["Smoking", "Peer Pressure", "Chronic Disease", "Alcohol Consuming"]
main_symptoms = ["Yellow Fingers", "Anxiety", "Fatigue", "Allergy", "Wheezing", "Coughing", "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain"]

# Features multiselect with relevant features
features = st.multiselect(
    "**Select Features**",
    options=main_features,
    default=main_features
)

# Symptoms multiselect based on symptom columns
symptoms = st.multiselect(
    "**Select Symptoms**",
    options=main_symptoms,
    default=main_symptoms
)

# Age slider based on the dataset's age range (1-120)
ages = st.slider(
    "**Select Age Range**", 
    min_value=1, 
    max_value=120, 
    value=(20, 50)
)

# Filter the dataframe based on widget inputs
lung_df_filtered = lung_df[
    (lung_df["Gender"].isin(genders)) &
    (lung_df["Age"].between(ages[0], ages[1])) 
]

lung_df_filtered = lung_df_filtered.sort_values(by="Age", ascending=True)

# Select only the necessary columns based on user input
columns_to_display = ["Age", "Gender", "Lung Cancer"] + features + symptoms
lung_df_filtered = lung_df_filtered[columns_to_display]

st.dataframe(
    lung_df_filtered,
    use_container_width=True,
    column_config={"Age": st.column_config.TextColumn("Age")},
)

# --------------------------------------------------------------
# LUNG CANCER STATISTICS

"---"

# Pie chart for Lung Cancer status
lung_cancer_counts = lung_df_filtered['Lung Cancer'].value_counts().reset_index()
lung_cancer_counts.columns = ['Status', 'Count']

lung_cancer_chart = (
    alt.Chart(lung_cancer_counts)
    .mark_arc(innerRadius=50, stroke='white')
    .encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Status", type="nominal", legend=alt.Legend(title="Lung Cancer Status")),
        tooltip=['Status', 'Count']
    )
    .properties(title="Lung Cancer Status Distribution")
)

st.altair_chart(lung_cancer_chart, use_container_width=True)

"---"

# Bar chart for Gender distribution
gender_counts = lung_df_filtered['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

gender_chart = (
    alt.Chart(gender_counts)
    .mark_bar()
    .encode(
        x=alt.X('Gender:N', title='Gender'),
        y=alt.Y('Count:Q', title='Count'),
        color='Gender:N',
        tooltip=['Gender', 'Count']
    )
    .properties(title="Gender Distribution")
)

st.altair_chart(gender_chart, use_container_width=True)

# --------------------------------------------------------------

# Streamlit UI
st.subheader("📋 Lung Cancer Prediction Survey")
st.write("**Enter the patient's information below to predict the likelihood of lung cancer:**")

@st.cache_data
def load_models():
    # Load the models
    lr_model = joblib.load('models/lr_model.pkl')  # Corrected model name
    knn_model = joblib.load('models/knn_model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return lr_model, knn_model, label_encoder, scaler

lr_model, knn_model, label_encoder, scaler = load_models()

# Age input
age = st.slider("**Select Age**", min_value=1, max_value=120, value=30)

# Gender selection
gender = st.selectbox("**Select Gender**", options=["Male", "Female"])

# User input for binary features
feature_inputs = {}
for feature in lung_df.columns:
    if feature not in columns and feature != "Lung Cancer":
        feature_inputs[feature] = st.selectbox(f"**{feature}?**", options=["No", "Yes"])

# Model selection
model_choice = st.selectbox("**Choose a Model for Prediction**", options=["Logistic Regression", "K-Nearest Neighbors"])
selected_model = lr_model if model_choice == "Logistic Regression" else knn_model

if st.button("Predict"):
    # Prepare input data for prediction
    input_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        **feature_inputs,
        "Lung Cancer": "No"
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if col not in columns:
            input_df[col] = label_encoder.transform(input_df[col])

    # Transform features
    del input_df["Lung Cancer"]  
    input_df = scaler.transform(input_df)

    # Prediction
    prediction = selected_model.predict(input_df)
    result = "Likely to have lung cancer." if prediction[0] == 1 else "Unlikely to have lung cancer."
    # Display prediction result
    st.write("\n\n**Prediction Result:**", result)

# --------------------------------------------------------------


# logo and images

UKZN_LOGO = "images/UKZN.png"
st.logo(
    UKZN_LOGO,
    icon_image=UKZN_LOGO,
    size="large"
)

# make logo vanish when scrolling down
st.markdown(
    """
    <style>
    img[data-testid="stLogo"] {
        height: 4rem;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------


"---"
st.subheader("🌍 Lung Cancer Research")
st.write(
    """
    **Research done by
    [**Dylan Govender**](mailto:221040222@stu.ukzn.ac.za) & [**Yuvika Singh**](mailto:SinghY1@ukzn.ac.za)
    at the [**University of KwaZulu-Natal**](https://ukzn.ac.za/)**
    """ 
)

"---"
st.subheader("🔗 References")

st.write(
    "- **[Lung Cancer DataSet](https://www.kaggle.com/datasets/yusufdede/lung-cancer-dataset), Yusuf Dede (2018)**"
)

st.write(
    "- **[Lung and Colon Cancer Histopathological Image Dataset (LC25000)](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data), Borkowski AA (2019)**"
)

st.write(
    "- **[The IQ-OTH/NCCD lung cancer dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/data), Alyasriy (2023)**"
)
