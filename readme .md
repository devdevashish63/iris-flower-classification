# 🌸 Iris Flower Classification – Machine Learning Project

This project demonstrates a complete machine learning pipeline using the famous [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/iris). The objective is to classify flowers into three species (*Iris Setosa, Versicolor, Virginica*) based on their petal and sepal dimensions.

---

## 🚀 Project Highlights

- ✅ Used **six classification algorithms**
- ✅ Applied **10-fold cross-validation** to evaluate models
- ✅ Compared model performance using **boxplots**
- ✅ Final model evaluated with **accuracy**, **confusion matrix**, and **classification report**

---

## 📊 Models Used

| Algorithm | Library |
|-----------|---------|
| Logistic Regression | `sklearn.linear_model` |
| Linear Discriminant Analysis | `sklearn.discriminant_analysis` |
| K-Nearest Neighbors (KNN) | `sklearn.neighbors` |
| Decision Tree Classifier | `sklearn.tree` |
| Naive Bayes | `sklearn.naive_bayes` |
| Support Vector Machine (SVM) | `sklearn.svm` |

---

## 📁 Dataset

- **Source:** UCI Machine Learning Repository  
- **Features:**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target:** Flower class (Setosa, Versicolor, Virginica)

---

## 🧠 Workflow

1. **Data Loading** – Read CSV data from the UCI repository  
2. **Data Splitting** – 80% training, 20% validation  
3. **Model Evaluation** – Used 10-fold Stratified K-Fold cross-validation  
4. **Model Comparison** – Visualized accuracy of each algorithm  
5. **Final Prediction** – Trained SVM and evaluated performance on validation data  

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

---

## 📈 Output Plot

A boxplot comparing the performance of all models:

> *(Include a screenshot of the boxplot here if possible, or generate one after running the script)*

---

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- scikit-learn

---

## 📌 How to Run

### Step 1: Clone the repo
```bash
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
