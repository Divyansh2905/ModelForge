# 🏆 ModelForge: 3rd Place Winning Solution

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-v4.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3-orange.svg)](https://scikit-learn.org/)
[![Hackathon](https://img.shields.io/badge/Event-Dakshh_Tech_Fest-success.svg)](https://dakshh-hitk.com/)

**Team Name:** Random Forest Rangers  
**Team Members:** Divyansh Chhajer (Lead) & Somshubhro Guha  
**Event:** [ModelForge Machine Learning Championship](https://www.kaggle.com/competitions/model-forge-dakshh-finale) (Dakshh Tech Fest, Heritage Institute of Technology, Kolkata)  
**Final Result:** 3rd Place (Final RMSE: 2.23443)  

## 📌 Project Objective
The objective of this competition was to predict the **CPU user-time percentage (`usr`)** based on multi-user computing environment telemetry, including system calls, memory activity, and I/O rates.

## 🚧 The Challenges
Working with deep operating system telemetry presented several critical challenges:
* **Data Sparsity:** A relatively small training dataset of approximately 6,000 rows.
* **Skewed Distributions:** Highly right-skewed metrics (e.g., exponential read/write ratios).
* **Missing Telemetry:** Null values in critical kernel parameters like `runqsz` (run queue size) and `freemem` (free memory).
* **Inconsistent Target Variance:** The target variable (`usr`) exhibited wildly shifting variance across different system states. For instance, `critical` systems maintained a tight standard deviation of just 1.19, whereas `low` states were highly chaotic with a standard deviation of 26.63.

## 🚀 Our Winning Solution: The Optimal Hybrid Architecture
Standard global regression models failed to capture the rigid boundaries and shifting variances of the dataset. We engineered an adaptive **"Optimal Hybrid"** pipeline to systematically conquer these topological challenges.

### 1. Lossless Algebraic Imputation
Instead of relying on statistical median imputation (which introduces geometric distortion/noise), we reverse-engineered the system's generation logic. We perfectly recovered missing memory data by transposing deterministic algebraic relationships. For example, to find missing queue sizes: 
`runqsz = memory_pressure * (freemem + 1)`

### 2. Defensive Feature Engineering
* **Log-Transformations:** Applied `np.log1p()` across all heavily right-skewed telemetry to prevent extreme tails from dominating the gradient updates.
* **Organic Polynomials:** Engineered scale-invariant features like `scall_per_queue` (system calls per run queue depth) to provide clean interaction signals without exploding gradients.

### 3. Segmented "Divide and Conquer" Modeling
Because the `usr` variance shifted drastically based on the categorical `system_state`, a single global model was mathematically insufficient. We built a targeted ensemble strategy:
* **Micro-Models for Tight Variances:** We trained highly regularized, dedicated LightGBM models strictly for the narrow `critical`, `high`, and `medium` states.
* **Global Model for High Variance:** For the chaotic `low` state, splitting the data would have starved the model of training rows. Instead, we trained a Global LightGBM on *all non-idle rows* across the dataset, maximizing data exposure while successfully mapping the variance.

### 4. Deterministic Post-Processing
We applied rigid physical system constraints to the model's final predictions:
* **Workload Overrides:** Hard-coded predictions to strictly `0` if `workload_type == 'idle_light'`.
* **Boundary Clipping:** Hard-clipped all predictions to remain within their known physical operating limits (e.g., forcing any 'critical' prediction to stay strictly between 95 and 99).

## 💻 How to Run
All architecture, feature engineering, and modeling code is contained within the main Jupyter Notebook.

1. Clone this repository.
2. Install dependencies (e.g., `lightgbm`, `scikit-learn`, `pandas`, `numpy`).
3. Change the paths of the datasets and Run the complete pipeline via the notebook:
   `ModelForge_Winning_Solution.ipynb`

---
*Developed by the Random Forest Rangers.*
