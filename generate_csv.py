import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker for realistic fake data
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define parameters
n_rows = 10000
tech_eras = ['Tech 1.0 (1970-1990)', 'Tech 2.0 (1991-2010)', 'Tech 3.0 (2011-present)']
regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania']
income_levels = ['Low', 'Middle', 'High']
education_levels = ['High School', 'Some College', 'Bachelor', 'Master', 'PhD']
age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
device_types = ['Desktop', 'Laptop', 'Tablet', 'Smartphone', 'Wearable', 'None']
purchase_categories = ['Electronics', 'Software', 'Services', 'Accessories', 'Media']

# Generate synthetic data
data = []
for _ in range(n_rows):
    tech_era = random.choice(tech_eras)
    region = random.choice(regions)
    income = random.choice(income_levels)
    education = random.choice(education_levels)
    age_group = random.choice(age_groups)

    # Tech adoption metrics (vary by era)
    if tech_era == tech_eras[0]:
        tech_adoption = np.random.normal(0.3, 0.1)
        digital_literacy = np.random.normal(0.4, 0.15)
    elif tech_era == tech_eras[1]:
        tech_adoption = np.random.normal(0.6, 0.1)
        digital_literacy = np.random.normal(0.7, 0.1)
    else:
        tech_adoption = np.random.normal(0.85, 0.08)
        digital_literacy = np.random.normal(0.9, 0.05)

    # Adjust for income and education
    tech_adoption += (0.1 if income == 'High' else -0.1 if income == 'Low' else 0)
    digital_literacy += (0.15 if education in ['Master', 'PhD'] else 0)

    # Cap values between 0 and 1
    tech_adoption = max(0, min(1, tech_adoption))
    digital_literacy = max(0, min(1, digital_literacy))

    # Device ownership (weighted by era)
    if tech_era == tech_eras[0]:
        devices = random.choices(device_types[-2:], weights=[0.7, 0.3], k=1)[0]
    elif tech_era == tech_eras[1]:
        devices = random.choices(device_types[:3], weights=[0.4, 0.4, 0.2], k=1)[0]
    else:
        devices = random.choices(device_types[:4], weights=[0.2, 0.3, 0.3, 0.2], k=1)[0]

    # Purchase behavior
    purchase_frequency = np.random.poisson(3 if income == 'High' else 2 if income == 'Middle' else 1)
    avg_spend = np.random.lognormal(mean=5 if income == 'High' else 4, sigma=0.5)

    # Tech anxiety (inverse to adoption)
    tech_anxiety = 1 - tech_adoption + np.random.normal(0, 0.1)
    tech_anxiety = max(0, min(1, tech_anxiety))

    # Social influence score
    social_influence = np.random.beta(2, 2)

    # Purchase category (varies by era)
    if tech_era == tech_eras[0]:
        category = random.choices(purchase_categories, weights=[0.6, 0.1, 0.1, 0.1, 0.1])[0]
    elif tech_era == tech_eras[1]:
        category = random.choices(purchase_categories, weights=[0.4, 0.3, 0.1, 0.1, 0.1])[0]
    else:
        category = random.choices(purchase_categories, weights=[0.3, 0.2, 0.3, 0.1, 0.1])[0]

    # Brand loyalty score
    brand_loyalty = np.random.beta(2, 2) if income == 'High' else np.random.beta(1.5, 2)

    # Online vs offline purchase ratio
    online_ratio = tech_adoption * 0.8 + np.random.normal(0, 0.1)
    online_ratio = max(0, min(1, online_ratio))

    data.append([
        fake.uuid4(),        # Customer ID
        tech_era,
        region,
        income,
        education,
        age_group,
        tech_adoption,
        digital_literacy,
        devices,
        purchase_frequency,
        avg_spend,
        tech_anxiety,
        social_influence,
        category,
        brand_loyalty,
        online_ratio
    ])

# Define column names
columns = [
    'customer_id',
    'tech_era',
    'region',
    'income_level',
    'education_level',
    'age_group',
    'tech_adoption_score',
    'digital_literacy_score',
    'primary_device',
    'annual_tech_purchases',
    'avg_spend_per_item',
    'tech_anxiety_score',
    'social_influence_score',
    'preferred_category',
    'brand_loyalty_score',
    'online_purchase_ratio'
]

# Create a DataFrame and save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('consumer_behavior_tech_revolutions.csv', index=False)

print("CSV file 'consumer_behavior_tech_revolutions.csv' has been generated successfully.")
