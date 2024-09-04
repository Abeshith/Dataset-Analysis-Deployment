## Streamlit Application for Data Exploration, Prediction, and Clustering

This Streamlit application allows users to upload CSV data files and perform various analyses, including data visualization, prediction using machine learning models, and clustering. The app is divided into four main sections: **Data Analysis**, **Compare Plots**, **Prediction**, and **Clustering**.

### Sections

1. ### Data Analysis
   - **Upload CSV File:** Users can upload a CSV file for analysis.
   - **Plot Selection:** Choose from a variety of plots, including:
     - Histogram
     - Box Plot
     - Scatter Plot
     - Line Plot
     - Area Plot
     - Density Plot
     - Violin Plot
     - Bar Plot
     - Heatmap
     - Pair Plot
     - Correlogram
     - Bubble Plot
     - Time Series Plot
   - **Categorical Features:** Categorical features are automatically encoded for plotting.
   - **Plot Libraries:** Plots are generated using Plotly or Seaborn/Matplotlib, depending on the selected plot type.

2. ### Compare Plots
   - **Side-by-Side Comparison:** Allows users to compare two different types of plots using selected features from the dataset.
   - **Plot Selection:** Users can select any two plots from the list provided in the Data Analysis section.

3. ### Prediction
   - **Upload CSV File:** Users can upload a CSV file for prediction tasks.
   - **Model Selection:** Choose from several machine learning models, including:
     - Linear Regression
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Gradient Boosting
     - XGBoost
   - **Feature and Target Selection:** Select the features and target variable for model training.
   - **Prediction Results:** The app displays predictions and evaluation metrics:
     - Regression: Mean Squared Error (MSE)
     - Classification: Accuracy, Confusion Matrix

4. ### Clustering
   - **Clustering Algorithms:** Users can apply clustering algorithms on the dataset:
     - K-Means
     - Agglomerative Clustering
     - DBSCAN
   - **Visualization:** Clusters are visualized using a scatter plot. At least two features must be selected for clustering.

### How to Run the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repository.git

2. Navigate to the application directory:

   ```bash
   cd your-repository

3. Install the required packages:

   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
