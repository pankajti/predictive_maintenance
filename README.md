
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

To understand time-dependent degradation, I created a **simulated vibration dataset** that mimics real-world machine failures.
Each 1-second window of vibration amplitudes is labeled with its corresponding RUL.

I explored two deep learning approaches:

* **LSTM (Long Short-Term Memory)**: Captures sequence dependencies in vibration signals lstm implementation is at : [LSTM Implementation](https://github.com/pankajti/predictive_maintenance/tree/main/predictive_maintenance/simulated/lstm)
* **Transformer** (in progress): Allows global attention across time for potentially better degradation modeling

Using these models, I trained an RUL regressor and evaluated its predictions on synthetic failure curves.

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
