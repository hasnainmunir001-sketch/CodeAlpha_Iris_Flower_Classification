# 🌸 Iris Flower Classification — Machine Learning Project

A complete, professional, end-to-end machine learning project that classifies
Iris flowers (**Setosa, Versicolor, Virginica**) based on their measurements
— and also works with **any classification dataset** you upload (e.g. from
Kaggle). Built with **Scikit-learn** and **Streamlit**.

🔗 **Live Demo:** _add your Streamlit Cloud link here after deployment_

---

## 📌 Features

- ✅ Works instantly with the built-in Iris dataset — no setup needed
- ✅ Upload **any CSV dataset** (Kaggle or otherwise) and classify it
- ✅ Trains and compares **6 ML algorithms**: Logistic Regression, KNN, SVM,
  Decision Tree, Random Forest, Naive Bayes
- ✅ Full evaluation: Accuracy, Precision, Recall, F1-score, Cross-validation,
  Confusion Matrix, Classification Report
- ✅ Exploratory Data Analysis: correlation heatmap, pairplots, box plots,
  class distribution
- ✅ Interactive prediction panel — enter measurements and get a live prediction
  with class probabilities
- ✅ Download the trained model as a `.pkl` file
- ✅ Clean, professional UI ready to demo to recruiters or a company

---

## 🗂️ Project Structure

```
iris-classification/
│
├── app.py                 # Main Streamlit application (the dashboard)
├── train_model.py         # Standalone CLI script to train & save a model
├── utils.py                # Shared ML helper functions (training, evaluation, prediction)
├── requirements.txt        # Python dependencies
├── .gitignore               # Files/folders Git should ignore
├── README.md                # Project documentation (this file)
│
├── data/
│   └── iris.csv             # Sample dataset (classic Iris dataset)
│
└── models/                  # Saved model artifacts (generated after training)
    ├── .gitkeep
    ├── best_model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── feature_names.pkl
    └── metadata.json
```

---

## 🧠 ML Concepts Used

| Concept | Where it's used |
|---|---|
| **Classification** | Predicting a categorical label (species) from numeric features |
| **Train/Test Split** | `train_test_split` with stratification to keep class balance |
| **Feature Scaling** | `StandardScaler` so distance-based models (KNN, SVM) work correctly |
| **Label Encoding** | Converting text class labels into numbers |
| **Cross-Validation** | 5-fold CV to check model stability, not just a single split |
| **Model Comparison** | Multiple algorithms trained and ranked by accuracy |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-score, Confusion Matrix |

---

## 💻 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/iris-classification.git
cd iris-classification
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Train a model from the command line
```bash
python train_model.py --data data/iris.csv --target species
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## 🚀 Step-by-Step: Upload This Project to GitHub

Yahan har step detail mein diya hai:

### Step 1 — GitHub par naya repository banayein
1. [github.com](https://github.com) par login karein.
2. Top-right par **"+"** icon → **"New repository"** par click karein.
3. Repository ka naam dein, e.g. `iris-flower-classification`.
4. **Public** select karein (agar recruiters/company ko dikhana hai).
5. **"Add a README file"** ko UNCHECK rakhein (kyunki hamare paas already README hai).
6. **Create repository** par click karein.

### Step 2 — Apne computer par Git install karein (agar nahi hai)
- [git-scm.com](https://git-scm.com/downloads) se download karein aur install karein.
- Terminal/CMD mein check karein: `git --version`

### Step 3 — Project folder ko Git repository banayein
Apne project folder (`iris-classification`) ke andar terminal khol kar yeh commands chalayein:

```bash
git init
git add .
git commit -m "Initial commit: Iris Flower Classification project"
```

### Step 4 — Local repo ko GitHub repo se connect karein
GitHub par repository banane ke baad wahan aapko ek URL milega, jese:
`https://github.com/<your-username>/iris-flower-classification.git`

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/iris-flower-classification.git
git push -u origin main
```

Bas! Ab aapka pura project GitHub par live hoga.

### Step 5 — Future changes push karna
Jab bhi code mein koi change karein:
```bash
git add .
git commit -m "Describe your change here"
git push
```

---

## ☁️ Step-by-Step: Deploy on Streamlit Community Cloud

### Step 1 — Confirm GitHub repo is ready
Make sure `app.py` aur `requirements.txt` aapke GitHub repo ke **root folder** mein hain (jese isi project mein hain).

### Step 2 — Streamlit Cloud par account banayein
1. [share.streamlit.io](https://share.streamlit.io) par jayein.
2. **"Sign up"** / **"Continue with GitHub"** par click karke apne GitHub account se login karein.

### Step 3 — Naya app deploy karein
1. Dashboard mein **"New app"** button par click karein.
2. **Repository** dropdown se apna repo select karein: `<your-username>/iris-flower-classification`.
3. **Branch**: `main`.
4. **Main file path**: `app.py`.
5. **"Deploy!"** par click karein.

### Step 4 — Wait for build
Streamlit Cloud automatically `requirements.txt` se sab libraries install karega. Yeh 1–3 minute le sakta hai.

### Step 5 — App live ho jayegi
Aapko ek public URL milega jese:
`https://<your-app-name>.streamlit.app`

Yeh link aap apne resume, GitHub README, ya LinkedIn par share kar sakte hain.

### Updating the deployed app
Jab bhi aap `main` branch par naya code push karenge, Streamlit Cloud **automatically redeploy** kar dega — kuch extra karne ki zaroorat nahi.

---

## 📊 Model Performance (on built-in Iris dataset)

| Model | Accuracy | F1-Score |
|---|---|---|
| Support Vector Machine | ~96.7% | ~96.7% |
| Naive Bayes | ~96.7% | ~96.7% |
| K-Nearest Neighbors | ~93.3% | ~93.3% |
| Logistic Regression | ~93.3% | ~93.3% |
| Decision Tree | ~93.3% | ~93.3% |
| Random Forest | ~90.0% | ~90.0% |

_Exact numbers vary slightly depending on the random seed and train/test split._

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** — model training & evaluation
- **Pandas / NumPy** — data handling
- **Matplotlib / Seaborn** — visualization
- **Streamlit** — web app / dashboard
- **Joblib** — model serialization

---

## 📄 License

This project is open-source and available under the MIT License. Feel free to
use it in your portfolio, resume, or as a base for other classification
projects.

---

## 🙋 Author

Built as a portfolio-ready data science project demonstrating classification
fundamentals: data preprocessing, model training, evaluation, and deployment.
