# Consumer Behavior Analytics Dashboard

## Overview

This project is an interactive dashboard that analyzes consumer behavior across different technological eras using Streamlit. The dashboard provides insights into:

- **Public Perception:** Visualizes how technology adoption changes over time.
- **Psychological Impact:** Explores the relationship between digital literacy and tech anxiety.
- **Socioeconomic Analysis:** Examines annual tech purchases across different income levels.
- **Market Sentiment Prediction:** Simulates predictions of tech adoption based on consumer attributes.

Designed with user-friendliness in mind, this dashboard includes plain-language explanations to help even users new to computer science understand the graphs and outputs.

## Installation

### Clone the Repository:
```bash
git clone https://github.com/NeelThomas02/Consumer-Behavior-Analytics-Dashboard
cd Consumer-Behavior-Analytics
```
Create a Virtual Environment:
```bash
python -m venv .venv
```
Activate the Virtual Environment:
On Windows:

```bash
.\.venv\Scripts\activate
```
On macOS/Linux:

```bash
source .venv/bin/activate
```
Install Required Packages:
```bash
pip install -r requirements.txt
```
Usage
Generate the Synthetic Data:
Run the following command to generate the CSV file:

```bash
python synthetic_data_generation.py
```
This creates consumer_behavior_tech_revolutions.csv in the data/ folder (or in the project root, as specified).

Run the Dashboard:
Start the Streamlit app by running:

```bash
streamlit run app.py
```
The app will open in your default web browser. Use the sidebar filters to refine the data and navigate through the various tabs.

Features
Interactive Visualizations:
Leverages Plotly for interactive charts with hover functionality, allowing users to see detailed information.

User Guidance:
Each tab includes expanders with plain-language explanations to help users understand how to read the charts and interpret the data.

Data Filtering & Download:
The sidebar offers interactive filters (by region, income level, and age group) and provides a download button to export the filtered dataset as CSV.

Market Sentiment Prediction:
A simulated prediction system takes consumer attributes as inputs and calculates a simple score to predict tech adoption. This can be replaced with a trained ML model for more accurate predictions.

Requirements
```bash
Python 3.x

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

Plotly

scikit-learn
```

(See the requirements.txt file for specific package versions.)

Contributing
Contributions are welcome! If you find bugs or have suggestions for improvements, please open an issue or submit a pull request.