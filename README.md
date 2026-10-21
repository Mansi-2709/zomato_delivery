# 🍽️ Zomato Delivery Intelligence: Geospatial Clustering & Delivery Time Prediction

**An end-to-end Machine Learning and Delivery Analytics project using Python, Geospatial Clustering, Operational Segmentation, XGBoost, Random Forest, Ridge Regression, and Streamlit.**

## 📌 Project Overview

Food delivery performance depends on more than the distance between a restaurant and a customer. Traffic congestion, weather conditions, vehicle condition, delivery workload, location, and order timing can all influence delivery time.

This project analyzes Zomato delivery data to understand these operational factors, identify geographical delivery zones, segment delivery operations into meaningful clusters, and predict estimated delivery time using machine learning.

The solution combines data preprocessing, geospatial analysis, operational clustering, regression modeling, hyperparameter tuning, and an interactive Streamlit application to turn raw delivery data into actionable operational insights.

**The primary objective is to help delivery operations teams understand where delays may occur, identify areas that may require operational improvements, and estimate delivery time under different delivery conditions.**

## 🎯 Business Objectives

- **Improve delivery-time estimation:** Predict delivery duration using relevant delivery, traffic, weather, vehicle, and order-related features.
- **Identify geographical delivery zones:** Group delivery locations into geographical clusters to understand spatial distribution and regional delivery patterns.
- **Analyze operational patterns:** Segment deliveries based on operational characteristics to identify recurring combinations of traffic conditions, vehicle conditions, and vehicle types.
- **Support fleet planning:** Help identify locations where vehicle-condition improvements or a different vehicle mix may warrant further investigation.
- **Understand delivery bottlenecks:** Explore the relationship between delivery time, travel distance, traffic, weather, multiple deliveries, and other operational factors.
- **Compare machine learning approaches:** Evaluate Ridge Regression, Random Forest Regressor, and XGBoost Regressor to identify a suitable model for delivery-time prediction.

## 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Data cleaning and preprocessing | Pandas, NumPy |
| Exploratory Data Analysis | Pandas, Plotly |
| Geospatial analysis | Latitude/longitude processing, HDBSCAN-based geographical clustering |
| Operational segmentation | Clustering and cluster profiling |
| Machine Learning | Ridge Regression, Random Forest Regressor, XGBoost Regressor |
| Model optimization | Hyperparameter tuning, cross-validation and model evaluation |
| Visualization | Plotly, interactive geographical maps |
| Application development | Streamlit |
| Model persistence | Joblib, if used for the saved prediction model |
| Development environment | Jupyter Notebook, GitHub |

## 📂 Project Structure

The repository contains the following project assets. Update the filenames below to match the actual names committed to GitHub.

```text
zomato_delivery/
│
├── app.py
│   └── Streamlit application for delivery analytics and prediction
│
├── Zomato2.csv
│   └── Original raw dataset before cleaning
│
├── cleaned_delivery_data.csv
│   └── Cleaned dataset with processed features and cluster labels
│
├── zomato_delivery_analysis.ipynb
│   └── End-to-end data preparation, clustering, model comparison,
│       hyperparameter tuning, and evaluation
│
├── xgb_delivery_time_model.joblib
│   └── Saved trained XGBoost model, if loaded by the application
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
    └── Project documentation
```

> **Note:** If `app.py` loads a previously trained model from a `.joblib` file, include that model file in the repository or provide another reliable way to access it. A saved model avoids retraining on every prediction request.

## 📊 Dataset Description

The project uses a food-delivery dataset containing information about delivery personnel, restaurant and customer locations, order timing, weather, traffic, vehicle characteristics, and delivery duration.

### Key features used

| Feature | Description |
|---|---|
| `Delivery_person_Age` | Age of the delivery person |
| `Delivery_person_Ratings` | Delivery-person rating |
| `distance_km` | Calculated distance between restaurant and delivery location |
| `multiple_deliveries` | Number of deliveries handled together |
| `Vehicle_condition` | Recorded vehicle-condition category |
| `Traffic` | Encoded traffic condition |
| `Time_Orderd_Hour` | Hour at which the order was placed |
| `weather_num` | Encoded weather condition |
| `order_num` | Encoded order type |
| `vehicle_num` | Encoded vehicle type |
| `fest_num` | Festival indicator |
| `city_num` | Encoded city category |
| `Day_of_Week_Num` | Encoded day of the week |
| `Time_taken (min)` | Target variable: delivery duration in minutes |

The dataset also includes geographical coordinates and cluster labels used for geographical visualization and operational analysis.

## 🧹 1. Data Cleaning and Preprocessing

The raw dataset was processed in the Jupyter Notebook before being used for clustering and machine learning.

The preprocessing workflow includes:

- **Missing-value treatment:** Addressed missing values in important fields, including delivery-person age, ratings, weather, traffic, city, order timing, and multiple-delivery information.
- **Repeated-person information:** Used available records for the same delivery person where appropriate when handling missing age values.
- **Rating imputation:** Used delivery-person-level information where available and an appropriate fallback for unresolved missing ratings.
- **Order-time processing:** Standardized inconsistent time formats and extracted the order hour for downstream analysis.
- **Missing order-time handling:** Used the recorded pickup time to estimate missing order times where applicable, based on the project's chosen time assumption.
- **Coordinate validation:** Identified and removed erroneous geographical coordinate records before performing geographical analysis.
- **Feature engineering:** Created distance and model-ready encoded features from the original delivery information.
- **Data validation:** Checked the prepared data for missing or invalid values before model training.

The cleaned dataset retains the processed features and cluster labels so the Streamlit application can use the prepared data directly.

**Why this matters:** Reliable data preparation is essential because invalid coordinates can distort geographical clusters, while inconsistent or improperly handled features can affect both model performance and the credibility of operational insights.

## 🗺️ 2. Geographical Clustering

The geographical clustering component groups delivery locations according to their spatial distribution.

HDBSCAN was used to identify geographical clusters from delivery coordinates. Unlike a fixed-cluster-count approach, HDBSCAN can identify clusters of varying densities and classify some observations as noise.

### What this analysis helps investigate

- Which geographical areas contain concentrated delivery activity?
- How are restaurant and customer locations distributed across delivery zones?
- Which areas may have longer average delivery distances or delivery durations?
- Where might delivery demand and operational capacity need further investigation?
- Which regions could be considered for future delivery-zone optimization?

### Business application

Geographical clusters can help a delivery company organize its analysis around meaningful delivery zones instead of treating every delivery as an isolated transaction.

For example, a zone with high delivery activity and longer average delivery times could be investigated for traffic bottlenecks, delivery-person availability, restaurant preparation delays, or vehicle allocation.

The clusters provide a spatial foundation for further investigation; they do not, by themselves, prove that a particular location is underperforming.

## 🚦 3. Operational Clustering and Cluster Profiling

Geographical clustering explains **where deliveries occur**. Operational clustering provides a complementary view of **how delivery conditions differ across groups of deliveries**.

The project uses operational cluster labels to examine combinations of characteristics such as:

- Traffic conditions: High, Medium, Low, and Jam
- Vehicle condition
- Vehicle type, including motorcycles, scooters, and electric scooters
- Delivery workload and multiple deliveries
- Delivery distance and delivery duration

The resulting clusters can be profiled using descriptive statistics and category distributions to understand their dominant operational characteristics.

### Examples of operational questions

- Which operational groups are associated with heavy traffic?
- Which groups have a higher concentration of vehicles in poorer condition?
- Which groups are dominated by motorcycles, scooters, or electric scooters?
- How do delivery times and distances vary between operational groups?
- Which combinations of traffic, vehicle type, and vehicle condition deserve further investigation?

### How this can support business decisions

| Operational finding | Potential business action |
|---|---|
| A geographical zone has consistently high delivery times | Investigate congestion, restaurant preparation time, and delivery-person availability |
| A cluster contains many deliveries under heavy traffic | Evaluate dispatch timing, route planning, and workload distribution |
| A cluster has a high share of vehicles in poorer condition | Prioritize vehicle maintenance or condition assessments |
| Different vehicle types show different patterns across comparable zones | Evaluate whether fleet allocation should vary by location |
| Certain operational groups have high delivery durations despite shorter distances | Investigate non-distance factors, including traffic, order batching, and operational delays |

These are decision-support opportunities rather than causal conclusions. Before changing fleet allocation, a company should validate the findings against sufficient data and, where possible, test proposed changes using controlled operational comparisons.

## 📈 4. Exploratory Data Analysis (EDA)

Exploratory analysis is used to understand the delivery dataset before and alongside model development.

The analysis focuses on the relationships between delivery duration and factors such as:

- Delivery distance
- Traffic conditions
- Weather conditions
- Delivery-person ratings
- Vehicle condition and vehicle type
- Multiple deliveries
- Order timing and day of the week
- Geographical and operational clusters

Interactive charts and maps make it easier to compare patterns, inspect distributions, and identify areas that deserve further investigation.

The purpose of EDA is not just to display charts, but to connect observed patterns to operational questions that a delivery business could investigate.

## 🤖 5. Machine Learning: Delivery Time Prediction

The prediction task is formulated as a **supervised regression problem**.

The model uses delivery and operational features to estimate the expected delivery duration in minutes.

**Target variable:** `Time_taken (min)`

### Models evaluated

| Model | Role in the project |
|---|---|
| Ridge Regression | Linear baseline with L2 regularization |
| Random Forest Regressor | Captures nonlinear patterns and feature interactions using an ensemble of decision trees |
| XGBoost Regressor | Gradient-boosted decision trees for modeling complex relationships in structured data |

Evaluating multiple algorithms provides a more informed model-selection process than relying on a single model by default.

### Model development workflow

1. Prepare the feature matrix and target variable.
2. Exclude records that cannot be used for training because of unresolved invalid or missing values.
3. Split the data into training and evaluation sets.
4. Train the Ridge Regression, Random Forest Regressor, and XGBoost Regressor models.
5. Perform hyperparameter tuning to identify promising configurations.
6. Compare model performance using appropriate regression metrics.
7. Select the preferred model based on evaluation results.
8. Save the selected trained model for use in the Streamlit application, where applicable.

### Hyperparameter tuning

Hyperparameter tuning was performed for the three regression approaches to investigate how different model configurations affect predictive performance.

The workflow considers model-specific settings, such as:

- Regularization strength for Ridge Regression
- Number of trees and tree-related parameters for Random Forest
- Learning rate, tree depth, and number of estimators for XGBoost

The final choice should be based on measured validation performance and generalization, rather than model complexity alone.

### Evaluation metrics

The notebook can report the following metrics for comparing the models:

- **MAE (Mean Absolute Error):** The average absolute difference between actual and predicted delivery times, measured in minutes.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily than MAE.
- **R² Score:** Indicates how much of the target's variance is explained by the model relative to a mean-prediction baseline.

Actual metric values and the selected model should be documented after confirming the results from the final notebook. No performance figures are claimed here without verified evaluation results.

## 🖥️ 6. Streamlit Application

The Streamlit application turns the analysis into an interactive interface for exploring delivery patterns and estimating delivery duration.

### Application modules

**1. Exploratory Data Analysis**

- Explore delivery data through interactive visualizations.
- Examine delivery-related variables and operational patterns.
- Investigate how distance, traffic, weather, and other factors relate to delivery duration.

**2. Geographical and Operational Clustering**

- View delivery locations on interactive maps.
- Explore geographical delivery zones.
- Inspect operational cluster characteristics.
- Compare cluster profiles to identify potential areas for operational improvement.

**3. Delivery Time Prediction**

- Enter delivery-related inputs such as distance, traffic, weather, vehicle type, vehicle condition, and order timing.
- Generate an estimated delivery duration using the trained regression model.
- Use the prediction as an indicative estimate rather than a guaranteed delivery time.

### Why Streamlit?

Streamlit allows the data science workflow to be presented as an interactive application, making the results easier to explore without requiring users to run individual notebook cells.

The application connects data preparation, visual analytics, clustering, and predictive modeling into a single user-facing workflow.

## 💡 7. Business Impact and Practical Applications

The project is designed to support several potential business use cases:

**Geographical planning:** Identify delivery zones that may require deeper investigation into demand, travel distance, or delivery capacity.

**Fleet optimization:** Use vehicle-type and vehicle-condition patterns to identify where maintenance, vehicle upgrades, or alternative fleet allocation may be worth evaluating.

**Delivery-time estimation:** Generate estimates that can support operational planning and help set more realistic expectations.

**Operational bottleneck analysis:** Identify combinations of traffic, workload, distance, and other factors associated with longer delivery durations.

**Data-driven decision-making:** Combine exploratory analysis, geographical clustering, operational segmentation, and regression modeling rather than relying on isolated charts or a prediction model alone.

The central value of the project is the integration of **where deliveries happen, which operational patterns characterize them, and how long deliveries are expected to take**.

## ⚙️ 8. Run the Project Locally

### Prerequisites

- Python 3.10 or another version compatible with the installed project dependencies
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/Mansi-2709/zomato_delivery.git
cd zomato_delivery
```

### Step 2: Install dependencies

If the repository includes `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by the application and notebook, such as:

```bash
pip install streamlit pandas numpy plotly scikit-learn xgboost hdbscan joblib
```

### Step 3: Check the project files

Ensure that the application can access the cleaned dataset and any required model artifact using valid paths.

If the application loads a saved XGBoost model, ensure that the model file is present at the path expected by `app.py`.

### Step 4: Launch the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to interact with the application.

### Step 5: Explore the notebook

Open `zomato.ipynb` in Jupyter Notebook or JupyterLab to review the data cleaning, geographical clustering, operational clustering, model training, hyperparameter tuning, and evaluation workflow.

## 🔍 9. Key Technical Considerations

- **Data quality:** Geographical analysis depends on valid coordinates and sensible distance calculations.
- **Feature consistency:** The same feature definitions and categorical encodings must be used during model training and prediction.
- **Cluster interpretation:** Cluster IDs are identifiers, not rankings. Their business meaning comes from profiling the characteristics of each cluster.
- **Model evaluation:** Performance must be measured on data that was not used to fit the model. If hyperparameter tuning is used, a separate final test set helps provide a less biased performance estimate.
- **Data leakage:** Any preprocessing steps that learn parameters from data should be fitted on the training set and then applied to validation and test sets.
- **Operational decisions:** Clusters and model predictions reveal patterns and estimates, not proof of causation. Fleet changes should be validated using appropriate operational data and experiments.

## 🚀 10. Future Improvements

Potential extensions include:

- Integrating a live traffic or routing API to improve distance and travel-time features.
- Adding cluster-level delivery KPIs and automated alerts for potentially underperforming zones.
- Comparing predicted and actual delivery durations to monitor model errors.
- Applying explainability techniques such as SHAP to understand the factors contributing to individual predictions.
- Evaluating fleet-allocation scenarios across geographical and operational clusters.
- Retraining the prediction model as new delivery records become available.
- Deploying the application with automated data and model versioning.

## 👩‍💻 About This Project

This project demonstrates an end-to-end application of data science to a practical logistics problem, combining data preprocessing, exploratory analysis, geographical analysis, clustering, regression, hyperparameter optimization, and application development.

Rather than treating machine learning as an isolated modeling exercise, the project connects analytical findings to potential operational decisions around delivery zones, fleet condition, vehicle allocation, and delivery-time estimation.

**Repository:** [Mansi-2709/zomato_delivery](https://github.com/Mansi-2709/zomato_delivery)

**Application:** [App](https://zomatodelivery01.streamlit.app)

---

*Disclaimer: This is an independent portfolio project using the available dataset. The findings and predictions are intended for analytical and educational purposes and should not be interpreted as verified Zomato operational results.*
