# Movie Revenue Clustering & Prediction via Exploratory Data Analysis

A comprehensive data mining and machine learning project focused on preprocessing, exploring, clustering, and predicting movie revenues using standard data science techniques and Deep Learning (Multilayer Perceptron).

---

## 📌 Project Overview

This repository contains an end-to-end data mining pipeline applied to a dataset of **45,466 movies** across 24 initial features. The project addresses key challenges in real-world data, such as high missingness and non-random zero values in critical financial attributes (`budget` and `revenue`).

Key objectives of this project:
1. **Data Preprocessing & Cleaning**: Handling missing values, feature selection, one-hot encoding for multi-label categories, and filtering non-random zero entries.
2. **Exploratory Data Analysis (EDA)**: Univariate, bivariate, and multivariate distribution and correlation analysis.
3. **Clustering Analysis**: Partitioning movies into optimal clusters using **K-Means** and **Agglomerative Hierarchical Clustering**, evaluated via **Elbow Method (Inertia)** and **Silhouette Scores**.
4. **Deep Learning Predictive Model**: Training a fully connected **Multilayer Perceptron (MLP)** network to predict box-office revenues based on extracted feature subsets and cluster assignments.

---

## 🛠️ Data Preprocessing & Cleaning Pipeline

The raw dataset contained 45,466 instances with 17 qualitative and 7 quantitative features. The preprocessing strategy followed a rigorous 7-step pipeline:

- **Step 1: Feature Elimination**: Removed non-informative variables including unique database identifiers (`id`, `imdb_id`), text overviews (`overview`), and poster paths (`poster_path`).
- **Step 2: Binary Transformation**: Converted high-missingness optional fields (`homepage`, `belongs_to_collection`, `tagline`) into binary presence indicators (1/0).
- **Step 3: Listwise Deletion**: Removed rows with under 1% total missing values (423 records).
- **Step 4: Multi-Category Expansion**: Encoded multi-label features:
  - `genres`: Converted into 20 binary indicators.
  - `production_countries`, `production_companies`, `spoken_languages`: Extracted top 10 most frequent categories per field plus an "Other" category (generating 33 binary features).
- **Step 5: Feature Engineering**: Extracted year, month, and day from `release_date`; simplified `status` to binary release indicators; removed redundant `original_language`.
- **Step 6: Memory & Type Optimization**: Cast variables to optimal data types for computational efficiency.
- **Step 7: Zero-Value Strategy**: Identified non-random zeros in financial figures. Constructed a refined subset of **5,380 verified movie entries** with full financial records for clustering and neural network regression.

---

## 📊 Exploratory Data Analysis (EDA)

- **Univariate Analysis**: Explored distributions using histograms and boxplots for `budget`, `runtime`, `vote_average`, `popularity`, and `revenue`. Strong right-skewness observed in `budget`, `popularity`, and `revenue`.
- **Bivariate Analysis**: Confirmed a strong linear relationship between `budget` and `revenue`. Analyzed non-linear patterns between `vote_average` and box-office outcomes.
- **Multivariate Analysis**: Correlation heatmap analysis indicated strong linear dependencies between `revenue`, `budget`, and `vote_count`.

---

## 🧩 Clustering Analysis

Movies were clustered using three key numeric features: `budget`, `runtime`, and `vote_average` (scaled via `StandardScaler`). `revenue` was excluded to prevent target leakage.

- **Algorithms Evaluated**:
  - **K-Means Clustering** ($k \in [2, 10]$)
  - **Agglomerative Hierarchical Clustering** ($k \in [2, 10]$)
- **Evaluation Metrics**: Inertia (Elbow Method) and Silhouette Score.
- **Optimal Cluster Count**: $k = 5$ provided the best trade-off between inertia reduction and silhouette score separation.
- **Post-Clustering Integration**: Cluster labels were one-hot encoded and incorporated into the feature matrix for downstream regression modeling.

---

## 🧠 Neural Network Model Architecture

A Fully Connected Multilayer Perceptron (MLP) was constructed to predict movie revenue:

### Data Split
- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

### Architecture & Hyperparameters
- **Input Layer**: Feature matrix normalized using training set statistics via `StandardScaler`. Target variable (`revenue`) standardized during training and inverse-transformed for evaluation.
- **Hidden Layer 1**: 64 neurons, ReLU activation function.
- **Hidden Layer 2**: 32 neurons, ReLU activation function.
- **Output Layer**: 1 neuron, Linear activation function.
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Training Epochs**: 30 epochs, Batch Size = 32

---

## 📈 Results & Evaluation

The predictive performance was evaluated on the unseen test dataset:

| Metric | Performance |
| :--- | :--- |
| **Coefficient of Determination ($R^2$)** | **~0.71** |
| **Loss Function** | Mean Squared Error (MSE) |

An $R^2$ score of **~0.71** demonstrates that the neural network successfully captures a significant portion of variance in movie revenues based on pre-production and descriptive features.

---

## 👨‍💻 Author

**Amin Namanalhosseini**  
Data Science Student  
Faculty of Mathematical Sciences, Shahid Beheshti University
