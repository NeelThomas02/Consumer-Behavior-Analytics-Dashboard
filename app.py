import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Set up page configuration (default dark theme is retained)
st.set_page_config(
    page_title="Consumer Behavior Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar filters for interactive data exploration
st.sidebar.header("Data Filters")
region_filter = st.sidebar.multiselect(
    "Select Regions", 
    options=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"],
    default=["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
)
income_filter = st.sidebar.multiselect(
    "Select Income Levels", 
    options=["Low", "Middle", "High"],
    default=["Low", "Middle", "High"]
)
age_filter = st.sidebar.multiselect(
    "Select Age Groups", 
    options=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    default=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
)

# Cache data using the new caching method
@st.cache_data
def load_data():
    df = pd.read_csv('consumer_behavior_tech_revolutions.csv')
    # Encode categorical features for visualization
    label_encoders = {}
    for col in ['tech_era', 'income_level', 'age_group']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    tech_era_labels = label_encoders['tech_era'].classes_
    income_level_labels = label_encoders['income_level'].classes_
    age_group_labels = label_encoders['age_group'].classes_
    df['tech_era_label'] = df['tech_era'].map(lambda x: tech_era_labels[x])
    df['income_level_label'] = df['income_level'].map(lambda x: income_level_labels[x])
    df['age_group_label'] = df['age_group'].map(lambda x: age_group_labels[x])
    return df

df = load_data()

# Filter data using the label columns directly
df_filtered = df[
    (df['region'].isin(region_filter)) &
    (df['income_level_label'].isin(income_filter)) &
    (df['age_group_label'].isin(age_filter))
]

# Provide a download button for the filtered data
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df_filtered)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv'
)

# Global Title and Introduction
st.title("Consumer Behavior Analytics Dashboard")
st.markdown(
    """
    Explore consumer behavior insights across different technological eras.
    Use the sidebar filters to refine the data, and navigate through the tabs below to view analyses on public perception,
    psychological impacts, socioeconomic inequalities, and sentiment-driven market predictions.
    """
)

# "How to Use" section as an expander
with st.expander("How to Use This Dashboard"):
    st.markdown("""
    **Overview:**  
    This dashboard analyzes consumer behavior across different technological eras with the following sections:  
    - **Public Perception:** Visualizes tech adoption and public sentiment.
    - **Psychological Impact:** Explores the relationship between digital literacy and tech anxiety.
    - **Socioeconomic Analysis:** Examines tech purchase behavior by income level.
    - **Market Sentiment Prediction:** Simulates predictions based on consumer attributes.
    
    **Navigation & Interaction:**  
    - Use the **sidebar filters** to select specific regions, income levels, and age groups.
    - Each **tab** represents a distinct analysis area.
    - Use the download button in the sidebar to export the filtered dataset.
    
    """)

# Use tabs for navigation between different panels
tabs = st.tabs([
    "Public Perception", 
    "Psychological Impact", 
    "Socioeconomic Analysis", 
    "Market Sentiment Prediction"
])

# Tab 1: Public Perception of Tech Surges
with tabs[0]:
    st.header("Public Perception of Tech Surges")
    st.markdown("Visualize how public sentiment and tech adoption have evolved over different eras.")

    with st.expander("How to Read This Chart"):
        st.markdown("""
        **Understanding the Box Plot:**

        - **X-Axis (Horizontal):**  
          This axis shows different time periods (called 'Tech Eras'):
          - **Tech 1.0 (1970-1990)**
          - **Tech 2.0 (1991-2010)**
          - **Tech 3.0 (2011-present)**
          
          These eras represent times when technology was first introduced and later improved.
          
        - **Y-Axis (Vertical):**  
          This axis shows the "tech adoption score."  
          - A **low score (closer to 0)** means fewer people were using new technology.
          - A **high score (closer to 1)** means more people adopted new technology.
          
        - **The Box Itself:**  
          - The box shows the middle range of the data (from the 25th to the 75th percentile).
          - The line inside the box is the **median**, which is the middle value.
          
        - **Whiskers and Outliers:**  
          - Lines extending from the box (whiskers) show the overall spread of the scores.
          - Points outside the whiskers are considered unusual values (outliers).
          
        - **Hover Information:**  
          - When you hover your mouse over a box, extra details (like region, income level, and age group) will appear.
          
        **What This Chart Tells Us:**  
        - In **Tech 1.0**, tech adoption scores are generally lower, meaning fewer people used new technology.  
        - In later eras (**Tech 2.0** and **Tech 3.0**), scores increase, showing that more people adopted technology over time.
        """)
    
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    order = ['Tech 1.0 (1970-1990)', 'Tech 2.0 (1991-2010)', 'Tech 3.0 (2011-present)']
    sns.boxplot(x='tech_era_label', y='tech_adoption_score', data=df_filtered, order=order, ax=ax1)
    ax1.set_title('Tech Adoption Scores by Technological Era', fontsize=14)
    ax1.set_xlabel('Technological Era', fontsize=12)
    ax1.set_ylabel('Tech Adoption Score', fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    st.info("Public sentiment improved from initial skepticism in Tech 1.0 to broad acceptance in Tech 3.0.")

# Tab 1: Psychological Impact on the Workforce
with tabs[1]:
    st.header("Psychological Impact on the Workforce")
    st.markdown("Analyze the relationship between digital literacy and tech anxiety across age groups.")

    with st.expander("How to read this interactive graph"):
        st.markdown("""
        **Scatter Plot Explanation:**
        - **X-Axis:** Digital Literacy Score (0-1) — how proficient a consumer is with technology.
        - **Y-Axis:** Tech Anxiety Score (0-1) — how much anxiety a consumer experiences when using technology.
        - **Data Points:** Each point represents a consumer or group of consumers.
        - **Color Coding:** Points are colored by Age Group, indicating differences in tech comfort across ages.
        - **Hover Details:** Hover over a point to see additional details like income level, region, and age group.
        
        **What to Look For:**
        - Typically, higher digital literacy (right side) is associated with lower tech anxiety (lower on the chart).
        - Notice if older age groups (indicated by color) tend to have higher anxiety.
        """)
    
    # Create an interactive Plotly scatter plot for hover functionality
    fig2 = px.scatter(
        df_filtered,
        x="digital_literacy_score",
        y="tech_anxiety_score",
        color="age_group_label",
        hover_data=["income_level_label", "region", "age_group_label"],
        title="Digital Literacy vs Tech Anxiety"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info("Hover over points to see more details about each consumer's data.")

# Tab 2: Socioeconomic Inequalities and Tech Waves
with tabs[2]:
    st.header("Socioeconomic Inequalities and Tech Waves")
    st.markdown("Examine how annual tech purchases vary by income level across technological eras.")

    with st.expander("How to read this graph"):
        st.markdown("""
        **Bar Chart Explanation:**
        - **X-Axis:** Income Level (Low, Middle, High) categorizes consumers by their economic status.
        - **Y-Axis:** Annual Tech Purchases — the number of tech products bought per year.
        - **Bars:** Each bar represents an income group, with colors showing different technological eras.
        
        **What to Look For:**
        - Compare the heights of the bars to see which income groups purchase more tech products.
        - Observe how the purchasing behavior changes across different tech eras.
        - This helps illustrate how technological advances might widen or bridge the economic divide.
        """)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    order_income = ['Low', 'Middle', 'High']
    order_era = ['Tech 1.0 (1970-1990)', 'Tech 2.0 (1991-2010)', 'Tech 3.0 (2011-present)']
    sns.barplot(
        x='income_level_label', 
        y='annual_tech_purchases', 
        hue='tech_era_label', 
        data=df_filtered, 
        order=order_income, 
        hue_order=order_era, 
        palette='viridis', 
        ax=ax3
    )
    ax3.set_title('Annual Tech Purchases by Income Level and Tech Era', fontsize=14)
    ax3.set_xlabel('Income Level', fontsize=12)
    ax3.set_ylabel('Annual Tech Purchases', fontsize=12)
    st.pyplot(fig3)
    st.info("The bar chart shows how tech purchases differ by income group over various tech eras.")


# Tab 4: Sentiment-Driven Market Predictions
with tabs[3]:
    st.header("Sentiment-Driven Market Predictions")
    st.markdown("Input consumer attributes below to simulate a prediction of the tech adoption category.")
    
    with st.expander("How to interpret this prediction"):
        st.markdown("""
        **What is happening here?**  
        This section uses your inputs to calculate a simple "score" that helps predict how likely a consumer is to adopt new technology.

        **Inputs Explained:**  
        - **Technological Era:**  
          Choose the time period (e.g., Tech 1.0, Tech 2.0, or Tech 3.0).  
          This tells us when the consumer is interacting with technology.
          
        - **Income Level and Age Group:**  
          These provide context about the consumer's background.
          
        - **Digital Literacy Score:**  
          A value from 0 to 1 that shows how familiar the consumer is with technology.  
          A higher score means the consumer is more comfortable with tech.
          
        - **Tech Anxiety Score:**  
          A value from 0 to 1 that indicates how much anxiety or stress the consumer feels about using new technology.  
          A higher score means more anxiety.
          
        - **Social Influence Score:**  
          A value from 0 to 1 that reflects how much the consumer's peers influence their tech decisions.
          
        - **Annual Tech Purchases and Average Spend:**  
          These numbers show how often and how much the consumer spends on tech products.

        **How the Prediction Works:**  
        The system calculates a simple score using this formula:  
        **Score = Digital Literacy - Tech Anxiety + (Annual Purchases / 10)**  
        
        Based on the score:  
        - If the score is **less than 0.5**, the prediction is **Low Adoption**.  
        - If the score is **between 0.5 and 1.0**, the prediction is **Medium Adoption**.  
        - If the score is **1.0 or higher**, the prediction is **High Adoption**.

        **Note:** This is a simulated prediction for demonstration purposes. In a real application, you would replace this simple calculation with a machine learning model for more accurate results.
        """)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            tech_era = st.selectbox("Technological Era", options=df['tech_era_label'].unique(), help="Select the time period (e.g., Tech 1.0, Tech 2.0, Tech 3.0).")
            income_level = st.selectbox("Income Level", options=df['income_level_label'].unique(), help="Select the consumer's income level.")
            age_group = st.selectbox("Age Group", options=df['age_group_label'].unique(), help="Select the consumer's age group.")
        with col2:
            digital_literacy = st.slider("Digital Literacy Score", min_value=0.0, max_value=1.0, value=0.5, help="A value between 0 and 1; higher means more comfortable with technology.")
            tech_anxiety = st.slider("Tech Anxiety Score", min_value=0.0, max_value=1.0, value=0.5, help="A value between 0 and 1; higher means more anxious about using technology.")
            social_influence = st.slider("Social Influence Score", min_value=0.0, max_value=1.0, value=0.5, help="A value between 0 and 1 that shows peer influence on tech decisions.")
        
        annual_purchases = st.number_input("Annual Tech Purchases", min_value=0, value=3, help="Enter how many tech products the consumer buys in a year.")
        avg_spend = st.number_input("Average Spend per Item", min_value=0.0, value=100.0, help="Enter the average amount spent on a tech product.")
        
        submitted = st.form_submit_button("Predict Adoption Category")
    
    if submitted:
        # Simulated prediction logic; replace with your ML model's prediction as needed.
        score = digital_literacy - tech_anxiety + (annual_purchases / 10)
        if score < 0.5:
            prediction = "Low Adoption"
        elif score < 1.0:
            prediction = "Medium Adoption"
        else:
            prediction = "High Adoption"
        st.success(f"Predicted Tech Adoption Category: {prediction}")
        st.info("Note: This is a simulated prediction. Replace with a real ML model for more accurate results.")

