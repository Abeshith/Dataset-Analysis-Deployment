import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

# Set the page title and layout
st.set_page_config(page_title="CSV Data Explorer", layout="wide")

# Function to handle categorical features
def handle_categorical(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    return df

# Function to plot data using Plotly or Seaborn
def plot_data(df, plot_type, feature=None, feature2=None):
    plot_size = dict(width=800, height=600)  # Standard plot size
    if plot_type == 'Histogram':
        fig = px.histogram(df, x=feature, title=f"Histogram of {feature}", **plot_size)
    elif plot_type == 'Box Plot':
        fig = px.box(df, y=feature, title=f"Box Plot of {feature}", **plot_size)
    elif plot_type == 'Scatter Plot':
        fig = px.scatter(df, x=feature, y=feature2, title=f"Scatter Plot of {feature} vs {feature2}", **plot_size)
    elif plot_type == 'Line Plot':
        fig = px.line(df, x=feature, y=feature2, title=f"Line Plot of {feature} vs {feature2}", **plot_size)
    elif plot_type == 'Area Plot':
        fig = px.area(df, x=feature, y=feature2, title=f"Area Plot of {feature} vs {feature2}", **plot_size)
    elif plot_type == 'Density Plot':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(df[feature], shade=True, ax=ax)
        ax.set_title(f"Density Plot of {feature}")
        fig = plt.gcf()
    elif plot_type == 'Violin Plot':
        fig = px.violin(df, y=feature, title=f"Violin Plot of {feature}", **plot_size)
    elif plot_type == 'Bar Plot':
        fig = px.bar(df, x=feature, y=feature2, title=f"Bar Plot of {feature} vs {feature2}", **plot_size)
    elif plot_type == 'Heatmap':
        fig = px.imshow(df.corr(), color_continuous_scale='RdBu_r', title="Correlation Heatmap", **plot_size)
    elif plot_type == 'Pair Plot':
        fig = sns.pairplot(df)
        plt.gcf().set_size_inches(8, 6)
        fig = plt.gcf()
    elif plot_type == 'Correlogram':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlogram")
        fig = plt.gcf()
    elif plot_type == 'Bubble Plot':
        fig = px.scatter(df, x=feature, y=feature2, size=feature2, color=feature, title=f"Bubble Plot of {feature} vs {feature2}", **plot_size)
    elif plot_type == 'Time Series Plot':
        fig = px.line(df, x=feature, y=feature2, title=f"Time Series Plot of {feature} vs {feature2}", **plot_size)
    return fig

# Function for regression and classification models
def run_model(df, model_type, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Decision Tree Regressor':
        model = DecisionTreeRegressor()
    elif model_type == 'Random Forest Regressor':
        model = RandomForestRegressor()
    elif model_type == 'Support Vector Regressor':
        model = SVR()
    elif model_type == 'Logistic Regression':
        model = LogisticRegression()
    elif model_type == 'Decision Tree Classifier':
        model = DecisionTreeClassifier()
    elif model_type == 'Random Forest Classifier':
        model = RandomForestClassifier()
    elif model_type == 'Support Vector Classifier':
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if model_type in ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'Support Vector Regressor']:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")
    else:
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Convert unique target values to strings
        target_names = [str(x) for x in df[target].unique()]
        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, target_names=target_names)
        st.text(report)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")

        # Plot confusion matrix using Plotly
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=target_names,
            y=target_names,
            colorscale='Viridis',
            showscale=True
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label"
        )
        st.plotly_chart(fig)

# Function for clustering models
def run_clustering(df, cluster_type, features, n_clusters=None):
    X = df[features]
    X = StandardScaler().fit_transform(X)

    if cluster_type == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif cluster_type == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif cluster_type == 'DBSCAN':
        model = DBSCAN()

    df['Cluster'] = model.fit_predict(X)

    # Scatter plot of clusters
    if len(features) >= 2:
        fig = px.scatter(df, x=features[0], y=features[1], color='Cluster', title=f"{cluster_type} Clustering")
    else:
        st.warning("At least two features are required for plotting clusters.")
        fig = None

    return fig, df

# Streamlit application
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Data Analysis", "Compare Plots", "Prediction", "Clustering"],
            icons=["file-earmark", "graph-up", "bar-chart", "cluster"],
            default_index=0
        )

    if selected == "Data Analysis":
        st.title("CSV Dataset Analysis")

        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Handle categorical features
            df = handle_categorical(df)

            # Drop null values
            df = df.dropna()

            st.write("### Data Preview")
            st.write(df.head())

            # Plot selection
            plot_type = st.selectbox("Select Plot Type", [
                "Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Area Plot",
                "Density Plot", "Violin Plot", "Bar Plot", "Heatmap", 
                "Pair Plot", "Correlogram", "Bubble Plot", "Time Series Plot"
            ])

            # Feature selection for plot
            if plot_type in ["Histogram", "Box Plot", "Density Plot", "Violin Plot"]:
                feature = st.selectbox("Select Feature", df.columns)
                if st.button("Generate Plot"):
                    fig = plot_data(df, plot_type, feature)
                    if isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                    else:
                        st.plotly_chart(fig)
            elif plot_type in ["Scatter Plot", "Line Plot", "Area Plot", "Bar Plot", 
                               "Bubble Plot", "Time Series Plot"]:
                feature1 = st.selectbox("Select Feature 1", df.columns)
                feature2 = st.selectbox("Select Feature 2", df.columns)
                if st.button("Generate Plot"):
                    fig = plot_data(df, plot_type, feature1, feature2)
                    if isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                    else:
                        st.plotly_chart(fig)
            else:
                if st.button("Generate Plot"):
                    fig = plot_data(df, plot_type)
                    if isinstance(fig, plt.Figure):
                        st.pyplot(fig)
                    else:
                        st.plotly_chart(fig)

    elif selected == "Compare Plots":
        st.title("Compare Plots")

        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Handle categorical features
            df = handle_categorical(df)

            # Drop null values
            df = df.dropna()

            # Select features for comparison
            feature1 = st.selectbox("Select First Feature", df.columns)
            feature2 = st.selectbox("Select Second Feature", df.columns)
            feature3 = st.selectbox("Select Third Feature", df.columns)
            feature4 = st.selectbox("Select Fourth Feature", df.columns)

            plot_type1 = st.selectbox("Select First Plot Type", [
                "Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Area Plot",
                "Density Plot", "Violin Plot", "Bar Plot", "Heatmap", 
                "Pair Plot", "Correlogram", "Bubble Plot", "Time Series Plot"
            ])
            plot_type2 = st.selectbox("Select Second Plot Type", [
                "Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Area Plot",
                "Density Plot", "Violin Plot", "Bar Plot", "Heatmap", 
                "Pair Plot", "Correlogram", "Bubble Plot", "Time Series Plot"
            ])

            # Generate plots
            if st.button("Generate Comparison Plots"):
                fig1 = plot_data(df, plot_type1, feature1, feature2)
                fig2 = plot_data(df, plot_type2, feature3, feature4)

                st.plotly_chart(fig1)
                st.plotly_chart(fig2)

    elif selected == "Prediction":
        st.title("Prediction")

        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Handle categorical features
            df = handle_categorical(df)

            # Drop null values
            df = df.dropna()

            # Select model type
            model_type = st.selectbox("Select Model Type", [
                'Linear Regression', 'Logistic Regression', 
                'Decision Tree Regressor', 'Decision Tree Classifier',
                'Random Forest Regressor', 'Random Forest Classifier',
                'Support Vector Regressor', 'Support Vector Classifier'
            ])

            # Select features and target
            features = st.multiselect("Select Features", df.columns)
            target = st.selectbox("Select Target", df.columns)

            # Run model
            if st.button("Run Model"):
                run_model(df, model_type, features, target)

    elif selected == "Clustering":
        st.title("Clustering")

        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Handle categorical features
            df = handle_categorical(df)

            # Drop null values
            df = df.dropna()

            # Select clustering type
            cluster_type = st.selectbox("Select Clustering Type", ['K-Means', 'Agglomerative', 'DBSCAN'])

            # Select features
            features = st.multiselect("Select Features", df.columns)

            # Select number of clusters for K-Means and Agglomerative
            n_clusters = None
            if cluster_type in ['K-Means', 'Agglomerative']:
                n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

            # Run clustering
            if st.button("Run Clustering"):
                fig, clustered_df = run_clustering(df, cluster_type, features, n_clusters)
                if fig is not None:
                    st.plotly_chart(fig)
                    st.write(clustered_df.head())

if __name__ == "__main__":
    main()
