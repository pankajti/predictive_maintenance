
---

# Predictive Maintenance — Personal Exploration & Prototype

## 📚 Learning Resources

To build foundational knowledge in predictive maintenance, I referred to the following key resources:

* **Book**: *An Introduction to Predictive Maintenance* (2nd Ed.) by R. Keith Mobley
* **Video Series**: [Adash Vibration Diagnostics YouTube Playlist](https://www.youtube.com/watch?v=BPMjYJ_HoWk&list=PLDNHqPpwBs8O2QIGHdi8Bwu3p-WbTLsXG)
* **Datasets**: NASA bearing dataset (via Kaggle) for hands-on experimentation

---

## 🎯 Problem Understanding

Predictive maintenance focuses on **forecasting machine failures before they occur**, using data like vibration signals, temperature, or pressure readings. 
The objective is to either:

* Predict **Remaining Useful Life (RUL)** of a component, or
* Detect **anomalies or failure conditions** ahead of time

Through vibration sensors, machines generate time-series data that reflects wear or evolving faults. By analyzing this data, we aim to**anticipate degradation trends**
and proactively plan maintenance.

---

## 🧠 Summary of Existing Approaches

Based on my understanding typical methods fall into three broad categories:

* **Statistical Feature Engineering**: Extracting time-window-based metrics such as mean, variance, skewness, and kurtosis to feed into traditional ML models
* **Classification or Anomaly Detection**: Labeling segments as “healthy,” “degraded,” or “failed,” often using random forests, SVMs, or autoencoders
* **RUL Regression**: Directly estimating the time until failure, typically using regression models or deep learning

While traditional approaches are valuable, many overlook the **temporal dynamics** inherent in sensor data. This is where sequence models become more powerful.
As per my understanding adopting an approach which retains the temporal nature of the data might be more useful and more aligned with the recent advancements 
in the AI models like Transformers which is the basis of many Large Language Models(LLMs). However, I also experimented with simpler models like LSTM for a 
gradual approach 

---

## 🔬 My Prototype and Experiments

### Introduction 

Remaining Useful Life (RUL) prediction is a critical task in predictive maintenance, particularly for complex systems like aircraft engines, where accurate forecasting can enhance safety, optimize maintenance schedules, and reduce operational costs. The NASA CMAPSS dataset, a widely used benchmark in prognostics, provides simulated turbofan engine degradation data, making it an ideal testbed for developing and evaluating RUL prediction models. This document outlines a novel approach for RUL prediction using a combination of Principal Component Analysis (PCA) and the TimesFM foundation model for time series forecasting, followed by a regression model to estimate RUL. The methodology leverages PCA to reduce the dimensionality of sensor and operational data, TimesFM to forecast future principal components, and a regression model to map these components to RUL values. The proposed workflow is evaluated on the CMAPSS FD001 dataset, with performance metrics including RMSE, R², and the NASA RUL score. The following sections detail the data processing, model architecture, implementation, and evaluation results, providing a comprehensive framework for accurate and efficient RUL prediction.

### Overall Strategy: PCA + TimesFM for Feature Forecasting + Regression for RUL

1. **Data Loading & RUL Calculation**

    * Data Loading: 
    
        Reads the train_FD001.txt, test_FD001.txt, and RUL_FD001.txt files. Make sure these files are in a CMAPSSData folder relative to your script, or adjust paths.
    
    * RUL Calculation:
    
        For the training data, it calculates the RUL for each cycle by subtracting the current time_in_cycles from the max_time_in_cycles for that specific engine.
    
    The RUL_THRESHOLD (e.g., 125) is applied to make the RUL piecewise linear. This is a common practice in CMAPSS literature: for very healthy early cycles, RUL is capped at a maximum value, as the model doesn't need to predict arbitrarily high RUL values, and the degradation signal is often not strong. This stabilizes training.


3. **Feature Selection**
   Select relevant sensor measurements and operational settings.

4. **Data Preprocessing (Scaling & PCA)**

   * Scale selected features using `MinMaxScaler`.
   * Apply **PCA** to reduce the dimensionality of the scaled features into a smaller set of **Principal Components (PCs)**.
   * The PCA model should be **fitted only on the training data**.

5. **TimesFM Forecasting of PCs**

   * For each engine's time series, get the historical sequence of PCs.
   * Use the **TimesFM** foundation model to forecast the future values of these PCs for a defined `horizon_len`.
   * TimesFM operates on **individual time series**, so you'll forecast each PC series separately.
   * ⚠️ **Important**: TimesFM is designed for general time series forecasting, not specifically for RUL. It will only predict future values of your PCs.

6. **Regression Model (PCs → RUL)**

   * Train a separate regression model (e.g., `RandomForestRegressor`) to map **current PC values** to **current RUL values**.
   * Use `(current_PCs, current_RUL)` pairs from the training data to learn this mapping.

7. **Prediction & Evaluation on Test Data**
   For each test engine:

   * Take its historical sequence of PCs.
   * Use **TimesFM** to forecast its PCs for the desired `horizon_len`.
   * Extract the **last forecasted PC vector** (or an aggregate, depending on how you define the prediction point).
   * Feed this vector into the trained **regression model** to predict the RUL.
   * Evaluate the predicted RUL against the **true RUL** using metrics like `RMSE`, `R²`, or **NASA score**.


 
---

## 🛠️ Application in Production

In a real-world setting, the same pipeline can be extended by:

* Replacing simulated streams with **live IoT sensor feeds**
* Continuously retraining or fine-tuning the model as more data is logged
* Deploying the model to make **real-time RUL predictions**, enabling intelligent maintenance scheduling

---

## 🧩 Learning for me

While my professional background is rooted in financial modeling and AI applications, this exercise has helped 
me **quickly ramp up on core concepts in engineering analytics**. 
I’m getting exposure and gaining confidence to apply my end-to-end ML system design experience to industrial settings, especially where time-series data, 
anomaly detection, or predictive forecasting is involved.

---
