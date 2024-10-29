import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

st.subheader("🔍 Exploring Lung Cancer")
st.write(
    """
    This section visualizes data from [Exploring Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer/data).
    The effectiveness of cancer prediction system helps the people to know their cancer risk with low cost and it also helps people to take the appropriate decision based on their cancer risk status. 
    The data is collected from the website online lung cancer prediction system. Just 
    click on the widgets below to explore!
    """
)

cancer_directory = "data/survey_lung_cancer.csv"
@st.cache_data
def load_data():
    lung_df = pd.read_csv(cancer_directory)
    lung_df.columns = lung_df.columns.str.replace('_', ' ').str.strip().str.title()
    return lung_df

lung_df = load_data()

# Mapping dictionary for binary columns (1: No, 2: Yes)
binary_mapping = {1: "No", 2: "Yes", "YES": "Yes", "NO": "No"}

# List of binary columns that need mapping
binary_columns = ["Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease",
                  "Fatigue", "Allergy", "Wheezing", "Alcohol Consuming", "Coughing",
                  "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain", "Lung Cancer"]

# Apply mapping to each binary column
for col in binary_columns:
    lung_df[col] = lung_df[col].map(binary_mapping)

# Mapping dictionary for gender
gender_mapping = {"M": "Male", "F": "Female"}
lung_df["Gender"] = lung_df["Gender"].map(gender_mapping)

# Gender selection with mapped values
genders = st.multiselect(
    "**Select Gender**",
    options=lung_df["Gender"].unique().tolist(),
    default=["Male", "Female"]
)

# Features multiselect with relevant features
features = st.multiselect(
    "**Select Features**",
    options=["Smoking", "Peer Pressure", "Chronic Disease", "Alcohol Consuming"],
    default=["Smoking", "Peer Pressure", "Chronic Disease", "Alcohol Consuming"]
)

# Symptoms multiselect based on symptom columns
symptoms = st.multiselect(
    "**Select Symptoms**",
    options=["Yellow Fingers", "Anxiety", "Fatigue", "Allergy", "Wheezing", "Coughing",
             "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain"],
    default=["Yellow Fingers", "Anxiety", "Fatigue", "Allergy", "Wheezing", "Coughing",
             "Shortness Of Breath", "Swallowing Difficulty", "Chest Pain"]
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
columns_to_display = ["Age", "Gender"] + features + symptoms + ["Lung Cancer"]
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

"---"

# Optional: Add more charts for other features
# Example: Bar chart for Smoking status
smoking_counts = lung_df_filtered['Smoking'].value_counts().reset_index()
smoking_counts.columns = ['Smoking Status', 'Count']

smoking_chart = (
    alt.Chart(smoking_counts)
    .mark_bar()
    .encode(
        x=alt.X('Smoking Status:N', title='Smoking Status'),
        y=alt.Y('Count:Q', title='Count'),
        color='Smoking Status:N',
        tooltip=['Smoking Status', 'Count']
    )
    .properties(title="Smoking Status Distribution")
)

st.altair_chart(smoking_chart, use_container_width=True)

# --------------------------------------------------------------



# --------------------------------------------------------------
"---"
st.subheader("🌍 Lung Cancer Research UKZN")
st.write(
    """
    **Research done by
    [**Dylan Govender**](mailto:221040222@stu.ukzn.ac.za) & [**Yuvika Singh**](mailto:SinghY1@ukzn.ac.za)
    at the [**University of KwaZulu-Natal**](https://ukzn.ac.za/)**
    """ 
)

# [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)]

