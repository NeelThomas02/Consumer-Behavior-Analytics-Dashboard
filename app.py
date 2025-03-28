import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

# Set up page configuration for the dashboard
st.set_page_config(
    page_title="Consumer Behavior Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model and scaler using caching
@st.cache_resource
def load_model():
    model = joblib.load("model/knn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Load the synthetic dataset for visualizations
@st.cache_data
def load_data():
    df = pd.read_csv('consumer_behavior_tech_revolutions.csv')
    # Create a target variable for visualization purposes
    df['tech_adoption_category'] = pd.cut(
        df['tech_adoption_score'],
        bins=[0, 0.4, 0.7, 1],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    return df

df = load_data()

# Sidebar filters for data visualization
st.sidebar.header("Data Filters")
region_filter = st.sidebar.multiselect(
    "Select Regions", 
    options=df['region'].unique(),
    default=list(df['region'].unique())
)
income_filter = st.sidebar.multiselect(
    "Select Income Levels", 
    options=df['income_level'].unique(),
    default=list(df['income_level'].unique())
)
age_filter = st.sidebar.multiselect(
    "Select Age Groups", 
    options=df['age_group'].unique(),
    default=list(df['age_group'].unique())
)

# Filter the dataset based on sidebar selections
df_filtered = df[
    (df['region'].isin(region_filter)) &
    (df['income_level'].isin(income_filter)) &
    (df['age_group'].isin(age_filter))
]

# Download button for filtered data
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
    Use the sidebar filters to refine the data and navigate through the tabs below to view analyses on:
    - Public Perception
    - Psychological Impact
    - Socioeconomic Analysis
    - Market Sentiment Prediction
    """
)

with st.expander("How to Use This Dashboard"):
    st.markdown("""
    **Overview:**  
    This dashboard provides analyses on:
    - **Public Perception:** Box plot of tech adoption scores.
    - **Psychological Impact:** Scatter plot of digital literacy vs tech anxiety.
    - **Socioeconomic Analysis:** Bar chart of annual tech purchases.
    - **Market Sentiment Prediction:** Input consumer attributes to get a predicted tech adoption category.
    
    **Navigation:**  
    - Use the sidebar filters to select specific regions, income levels, and age groups.
    - Click the tabs to switch between different analyses.
    - Hover over interactive charts for more information.
    - Download filtered data from the sidebar.
    """)

#  tabs for different sections
tabs = st.tabs([
    "Public Perception", 
    "Psychological Impact", 
    "Socioeconomic Analysis", 
    "Market Sentiment Prediction"
])

# --- Tab 1: Public Perception ---
with tabs[0]:
    st.header("Public Perception of Tech Surges")
    st.markdown("This interactive box plot shows how technology adoption scores vary across different technological eras.")
    
    with st.expander("How to read this chart"):
        st.markdown("""
        **Box Plot Explanation:**
        - **X-Axis:** Tech eras (Tech 1.0, Tech 2.0, Tech 3.0).
        - **Y-Axis:** Tech Adoption Score (0 to 1).
        - **Box Elements:** The middle 50% of data with the median indicated.
        - **Whiskers:** Indicate variability outside the middle range.
        - **Hover:** Hover over the chart to see details like region, income level, and age group.
        """)
    
    fig1 = px.box(
        df_filtered,
        x="tech_era",
        y="tech_adoption_score",
        title="Tech Adoption Scores by Technological Era",
        category_orders={"tech_era": ['Tech 1.0 (1970-1990)', 'Tech 2.0 (1991-2010)', 'Tech 3.0 (2011-present)']},
        hover_data=["region", "income_level", "age_group"]
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.info("Hover over the boxes for detailed insights.")

# --- Tab 2: Psychological Impact ---
with tabs[1]:
    st.header("Psychological Impact on the Workforce")
    st.markdown("This interactive scatter plot shows the relationship between digital literacy and tech anxiety.")
    
    with st.expander("How to read this graph"):
        st.markdown("""
        **Scatter Plot Explanation:**
        - **X-Axis:** Digital Literacy Score (0 to 1).
        - **Y-Axis:** Tech Anxiety Score (0 to 1).
        - **Color:** Points are colored by age group.
        - **Hover:** Hover over points to see details like region and income level.
        """)
    
    fig2 = px.scatter(
        df_filtered,
        x="digital_literacy_score",
        y="tech_anxiety_score",
        color="age_group",
        hover_data=["region", "income_level", "age_group"],
        title="Digital Literacy vs Tech Anxiety"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info("Typically, higher digital literacy is associated with lower tech anxiety.")

# --- Tab 3: Socioeconomic Analysis ---
with tabs[2]:
    st.header("Socioeconomic Inequalities and Tech Waves")
    st.markdown("This bar chart compares annual tech purchases across income levels and tech eras.")
    
    with st.expander("How to read this chart"):
        st.markdown("""
        **Bar Chart Explanation:**
        - **X-Axis:** Income Level (Low, Middle, High).
        - **Y-Axis:** Annual Tech Purchases.
        - **Color:** Bars are colored by tech era.
        - **Interpretation:** Higher bars indicate more tech purchases.
        """)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    order_income = ['Low', 'Middle', 'High']
    order_era = ['Tech 1.0 (1970-1990)', 'Tech 2.0 (1991-2010)', 'Tech 3.0 (2011-present)']
    sns.barplot(
        x='income_level',
        y='annual_tech_purchases',
        hue='tech_era',
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
    st.info("Higher income groups tend to make more tech purchases, especially in later eras.")

# --- Tab 4: Market Sentiment Prediction ---
with tabs[3]:
    st.header("Market Sentiment Prediction")
    st.markdown("Input consumer attributes below to get a prediction for tech adoption using a trained model.")
    
    with st.expander("How to interpret this prediction"):
        st.markdown("""
        **Prediction Explanation:**
        - **Digital Literacy Score:** Indicates how comfortable the consumer is with technology.
        - **Tech Anxiety Score:** Reflects the consumer's nervousness or resistance toward using technology.
        - **Annual Tech Purchases:** The number of tech products purchased per year.
        - **Average Spend per Item:** The average amount spent on each tech product.
        
        **Expected Outputs:**
        - **Low Adoption:**  
          Likely when the consumer has:
          - Low digital literacy (e.g., around 0.2–0.4)
          - High tech anxiety (e.g., around 0.7–0.9)
          - Low annual tech purchases (e.g., 0 or 1 per year)
          - Low average spend (e.g., below 50)
          
        - **Medium Adoption:**  
          Likely when the consumer has:
          - Moderate digital literacy (e.g., around 0.5–0.7)
          - Moderate tech anxiety (e.g., around 0.5–0.6)
          - Moderate annual tech purchases (e.g., 2–3 per year)
          - Moderate average spend (e.g., around 100)
          
        - **High Adoption:**  
          Likely when the consumer has:
          - High digital literacy (e.g., above 0.8)
          - Low tech anxiety (e.g., below 0.3)
          - Frequent annual tech purchases (e.g., 4 or more per year)
          - High average spend (e.g., above 150)
        
        **Example Scenarios:**
        - **Low Adoption Example:**  
          Digital Literacy: 0.3, Tech Anxiety: 0.8, Annual Tech Purchases: 1, Average Spend: 50  
          → Likely predicted as **Low Adoption**
          
        - **High Adoption Example:**  
          Digital Literacy: 0.9, Tech Anxiety: 0.2, Annual Tech Purchases: 5, Average Spend: 200  
          → Likely predicted as **High Adoption**
        """)
    
    with st.form("prediction_form"):
        digital_literacy = st.slider("Digital Literacy Score", min_value=0.0, max_value=1.0, value=0.5, help="Higher means more comfortable with technology.")
        tech_anxiety = st.slider("Tech Anxiety Score", min_value=0.0, max_value=1.0, value=0.5, help="Higher means more anxious about technology.")
        annual_purchases = st.number_input("Annual Tech Purchases", min_value=0, value=3, help="Enter the number of tech products purchased annually.")
        avg_spend = st.number_input("Average Spend per Item", min_value=0.0, value=100.0, help="Enter the average amount spent on a tech product.")
        submitted = st.form_submit_button("Predict Adoption Category")
    
    if submitted:
        # Create an input vector for the 4 features (order must match training)
        input_features = np.array([[digital_literacy, tech_anxiety, annual_purchases, avg_spend]])
        # Scale the input using the loaded scaler
        input_scaled = scaler.transform(input_features)
        # Predict using the trained KNN model
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Tech Adoption Category: {prediction[0]}")

