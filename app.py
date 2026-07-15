"""
app.py
------
Iris Flower Classification — Professional Streamlit Dashboard.

Features:
    - Works out-of-the-box with the built-in Iris dataset
    - Also accepts ANY CSV dataset (e.g. downloaded from Kaggle) for
      generic multi-class classification
    - Lets the user pick the target column and feature columns
    - Trains and compares multiple ML algorithms
    - Shows accuracy, precision, recall, F1, confusion matrix, classification report
    - Interactive prediction panel for new samples
    - Clean, professional UI

Run locally:
    streamlit run app.py
"""

import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from utils import (
    get_numeric_feature_columns,
    compare_all_models,
    train_single_model,
    predict_single_sample,
    MODEL_REGISTRY,
)

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #4a2c6d;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6b6b6b;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f7f2fb;
        border: 1px solid #e6dcf0;
        border-radius: 10px;
        padding: 12px 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    .footer-note {
        color: #9a9a9a;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌸 Iris Flower Classification</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">A professional machine learning dashboard for classifying '
    'flower species — works with the built-in Iris dataset or any CSV you upload.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — data source & configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    data_source = st.radio(
        "Choose data source",
        ["Built-in Iris dataset", "Upload my own CSV (e.g. from Kaggle)"],
    )

    uploaded_file = None
    if data_source == "Upload my own CSV (e.g. from Kaggle)":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.divider()
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", value=42, step=1)


@st.cache_data
def load_builtin_iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    df["species"] = [iris.target_names[i] for i in iris.target]
    return df


# --------------------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------------------
if data_source == "Built-in Iris dataset":
    df = load_builtin_iris()
    default_target = "species"
else:
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file from the sidebar to get started, "
                "or switch to the built-in Iris dataset.")
        st.stop()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()
    default_target = df.columns[-1]

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar — column selection (depends on loaded df)
# --------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("🎯 Columns")
    target_col = st.selectbox(
        "Target / label column",
        options=list(df.columns),
        index=list(df.columns).index(default_target) if default_target in df.columns else len(df.columns) - 1,
    )

    numeric_feats_all = get_numeric_feature_columns(df, target_col)
    feature_cols = st.multiselect(
        "Feature columns (numeric)",
        options=numeric_feats_all,
        default=numeric_feats_all,
    )

    st.divider()
    model_choice = st.selectbox(
        "Model for prediction panel",
        options=["Best model (auto)"] + list(MODEL_REGISTRY.keys()),
    )

if not feature_cols:
    st.warning("Please select at least one numeric feature column from the sidebar.")
    st.stop()

if df[target_col].nunique() < 2:
    st.warning("The target column needs at least 2 distinct classes for classification.")
    st.stop()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_eda, tab_models, tab_predict = st.tabs(
    ["📊 Data Overview", "🔍 Exploratory Analysis", "🤖 Model Training & Evaluation", "🔮 Make a Prediction"]
)

# ============================== TAB 1: OVERVIEW ==============================
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", f"{df.shape[0]:,}")
    col2.metric("Total Features", f"{len(feature_cols)}")
    col3.metric("Classes", f"{df[target_col].nunique()}")
    col4.metric("Missing Values", f"{int(df[feature_cols + [target_col]].isna().sum().sum())}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Class Distribution")
    class_counts = df[target_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, ax=ax, palette="viridis")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    with st.expander("Statistical Summary"):
        st.dataframe(df[feature_cols].describe(), use_container_width=True)

# ============================== TAB 2: EDA ==============================
with tab_eda:
    st.subheader("Feature Correlation Heatmap")
    if len(feature_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[feature_cols].corr(), annot=True, cmap="magma", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select at least 2 features to see a correlation heatmap.")

    st.subheader("Pairwise Feature Relationships")
    if len(feature_cols) >= 2:
        sample_df = df[feature_cols + [target_col]].dropna()
        if len(sample_df) > 500:
            sample_df = sample_df.sample(500, random_state=42)
        pair_fig = sns.pairplot(sample_df, hue=target_col, palette="husl", corner=True)
        st.pyplot(pair_fig)
    else:
        st.info("Select at least 2 features to see pairwise plots.")

    st.subheader("Feature Distribution by Class")
    feat_to_plot = st.selectbox("Choose a feature", options=feature_cols)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(data=df, x=target_col, y=feat_to_plot, ax=ax, palette="Set2")
    st.pyplot(fig)

# ============================== TAB 3: MODEL TRAINING ==============================
with tab_models:
    st.subheader("Train & Compare Multiple Algorithms")
    st.caption(
        "Logistic Regression, K-Nearest Neighbors, Support Vector Machine, "
        "Decision Tree, Random Forest, and Naive Bayes are all trained and compared automatically."
    )

    run_button = st.button("🚀 Train Models", type="primary")

    if run_button or "comparison_df" in st.session_state:
        if run_button:
            with st.spinner("Training models..."):
                comparison_df, all_results = compare_all_models(
                    df, target_col, feature_cols, test_size=test_size, random_state=int(random_state)
                )
            st.session_state["comparison_df"] = comparison_df
            st.session_state["all_results"] = all_results

        comparison_df = st.session_state["comparison_df"]
        all_results = st.session_state["all_results"]

        best_row = comparison_df.iloc[0]
        st.success(f"✅ Best model: **{best_row['Model']}** with "
                   f"**{best_row['Accuracy']*100:.2f}%** test accuracy")

        st.subheader("Model Comparison Table")
        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                "Recall": "{:.2%}", "F1 Score": "{:.2%}", "CV Mean": "{:.2%}"
            }).background_gradient(subset=["Accuracy"], cmap="Greens"),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.barplot(data=comparison_df, x="Accuracy", y="Model", ax=ax, palette="crest")
        ax.set_xlim(0, 1)
        st.pyplot(fig)

        st.divider()
        st.subheader("Detailed Evaluation for a Specific Model")
        selected_for_detail = st.selectbox(
            "Choose a model to inspect", options=comparison_df["Model"].tolist()
        )
        detail = all_results[selected_for_detail]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(
                detail["confusion_matrix"], annot=True, fmt="d", cmap="Purples",
                xticklabels=detail["class_names"], yticklabels=detail["class_names"], ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with c2:
            st.markdown("**Classification Report**")
            report_df = pd.DataFrame(detail["report"]).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

        # Save best model artifacts in session for the prediction tab / download
        st.session_state["trained_artifacts"] = all_results
        st.session_state["best_model_name"] = best_row["Model"]

        st.divider()
        st.subheader("⬇️ Download Trained Model")
        dl_choice = st.selectbox(
            "Model to download", options=comparison_df["Model"].tolist(), key="dl_choice"
        )
        buf = io.BytesIO()
        joblib.dump({
            "model": all_results[dl_choice]["model"],
            "scaler": all_results[dl_choice]["scaler"],
            "label_encoder": all_results[dl_choice]["label_encoder"],
            "feature_cols": all_results[dl_choice]["feature_cols"],
        }, buf)
        st.download_button(
            "Download model (.pkl)", data=buf.getvalue(),
            file_name=f"{dl_choice.replace(' ', '_').lower()}_model.pkl",
            mime="application/octet-stream",
        )
    else:
        st.info("Click **Train Models** to start training and evaluating.")

# ============================== TAB 4: PREDICTION ==============================
with tab_predict:
    st.subheader("Predict the Species of a New Sample")

    if "trained_artifacts" not in st.session_state:
        st.warning("⚠️ Please train the models first in the **Model Training & Evaluation** tab.")
    else:
        all_results = st.session_state["trained_artifacts"]
        chosen_model_name = (
            st.session_state["best_model_name"] if model_choice == "Best model (auto)" else model_choice
        )
        if chosen_model_name not in all_results:
            chosen_model_name = st.session_state["best_model_name"]

        artifacts = all_results[chosen_model_name]
        st.caption(f"Using model: **{chosen_model_name}**")

        st.markdown("Enter feature values below:")
        input_values = []
        cols = st.columns(min(len(feature_cols), 4) or 1)
        for i, feat in enumerate(feature_cols):
            col_min = float(df[feat].min())
            col_max = float(df[feat].max())
            col_mean = float(df[feat].mean())
            with cols[i % len(cols)]:
                val = st.slider(feat, min_value=col_min, max_value=col_max, value=col_mean)
            input_values.append(val)

        if st.button("🔮 Predict Species", type="primary"):
            label, proba = predict_single_sample(
                artifacts["model"], artifacts["scaler"], artifacts["label_encoder"], input_values
            )
            st.success(f"### Predicted class: **{label}**")

            if proba is not None:
                proba_df = pd.DataFrame({
                    "Class": artifacts["class_names"],
                    "Probability": proba,
                }).sort_values("Probability", ascending=False)
                st.markdown("**Prediction Confidence**")
                st.dataframe(
                    proba_df.style.format({"Probability": "{:.2%}"}).background_gradient(
                        subset=["Probability"], cmap="Purples"
                    ),
                    use_container_width=True, hide_index=True,
                )
                fig, ax = plt.subplots(figsize=(6, 2.5))
                sns.barplot(data=proba_df, x="Probability", y="Class", ax=ax, palette="mako")
                ax.set_xlim(0, 1)
                st.pyplot(fig)

st.markdown(
    '<p class="footer-note">Built with Scikit-learn & Streamlit • '
    'Iris Flower Classification Project</p>',
    unsafe_allow_html=True,
)
