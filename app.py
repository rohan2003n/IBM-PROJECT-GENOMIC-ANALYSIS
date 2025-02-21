import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def preprocess_data(df):
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    df.dropna(axis=1, how='all', inplace=True)
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])
    
    return df, numerical_cols, categorical_cols

def apply_pca_tsne(df, numerical_cols):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    pca = PCA(n_components=min(10, scaled_data.shape[1]))
    pca_result = pca.fit_transform(scaled_data)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_result = tsne.fit_transform(scaled_data)
    
    return pca_result, tsne_result, pca

def train_model(df, numerical_cols):
    X = df[numerical_cols]
    y = df['gene_biotype'] if 'gene_biotype' in df.columns else df.iloc[:, -1]
    
    chi2_selector = SelectKBest(score_func=chi2, k=min(10, len(numerical_cols)))
    X_selected = chi2_selector.fit_transform(X, y)
    selected_features = X.columns[chi2_selector.get_support()]
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return accuracy, report, selected_features

st.title("Genomic Data Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your genomic dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, dtype=str, low_memory=False)
    df, numerical_cols, categorical_cols = preprocess_data(df)
    
    st.write("### Processed Data Preview")
    st.dataframe(df.head())
    
    st.write("### PCA & t-SNE Visualization")
    pca_result, tsne_result, pca = apply_pca_tsne(df, numerical_cols)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    ax[0].set_xlabel('Number of Components')
    ax[0].set_ylabel('Cumulative Explained Variance')
    ax[0].set_title('PCA Explained Variance')
    
    tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
    sns.scatterplot(x=tsne_df['t-SNE1'], y=tsne_df['t-SNE2'], ax=ax[1])
    ax[1].set_xlabel('t-SNE Component 1')
    ax[1].set_ylabel('t-SNE Component 2')
    ax[1].set_title('t-SNE Visualization')
    
    st.pyplot(fig)
    
    st.write("### Model Training & Evaluation")
    accuracy, report, selected_features = train_model(df, numerical_cols)
    
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
    st.write("**Selected Features:**", list(selected_features))
    st.write("**Classification Report:**")
    st.json(report)
