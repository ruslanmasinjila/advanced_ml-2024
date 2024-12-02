
#  SENSOR FUSION USING MACHINE LEARNING TECHNIQUES

#  Project Report By Ruslan Masinjila (202488331) Of Group 24


#  Course: COMP-6936 Advanced Machine Learning
#  Instructor: Dr. Lourdes Peña-Castillo


#  Memorial University of Newfoundland
#  St. Johns

#  December 2, 2024

---

# 1.	Introduction
<br>
Sensor fusion is the integration of homogenous or heterogeneous data and the associated errors/uncertainties from multiple sensors to produce more accurate and reliable information compared to the individual sensors. This technique is essential in various applications, including robotics, healthcare, and environmental sensing, where combining diverse data sources enhances system performance and decision-making. For example, in robotics, sensor fusion combines data from cameras, LiDAR, gyroscopes, and accelerometers for accurate and precise localization and navigation in dynamic environments. Current sensor fusion techniques include deterministic methods like the Kalman (KF), Extended Kalman (EKF), and Particle Filters (PF). These filters rely on mathematical models and assumptions about sensor noise and system dynamics. Probabilistic methods such as Bayesian networks also provide a framework for combining uncertain data. Despite their effectiveness in many scenarios, these approaches face limitations, such as sensitivity to model inaccuracies, difficulty handling large volumes of heterogeneous data, and challenges adapting to non-linear and complex environments. Additionally,  filters such as the EKF are prone to producing inconsistent error estimates due to over-convergence or divergence of fused uncertainties.
Machine learning (ML) techniques offer promising solutions to overcome these limitations because they can learn complex patterns and relationships between sensor data without relying on explicit mathematical models. Techniques such as deep learning are increasingly used for sensor fusion. For instance, convolutional neural networks (CNNs) can process spatial data, such as depth data from various sources for robot localization and mapping. On the other hand, recurrent neural networks (RNNs) can analyze temporal sensor data for trajectory prediction. These ML-based approaches have demonstrated superior performance in non-linear and high-dimensional data scenarios. However, ML models often require extensive labeled data for training, which can be costly and time-consuming to acquire. Additionally, ensuring the real-time performance and interpretability of ML-based fusion systems is a significant challenge, particularly in safety-critical applications like autonomous driving. 

---

## 1.1 Problem Statement
This project aims to study the feasibility of ML approaches in sensor fusion. Specifically, the project investigates the performance of Generalized Additive Models (GAM), Generative Adversarial Networks (GAN), and Convolutional Neural Networks (CNN) in the fusion of simulated range-bearing sensor data for robot localization.
1.1.1	Specific Objectives
The specific objectives of this project are as follows:
Generate a synthetic dataset simulating range-bearing sensor measurements with random uncertainty.
Divide the dataset into training and validation sets.
Train and evaluate the performance of Generalized Additive Models (GAM), Generative Adversarial Networks (GAN), and Convolutional Neural Networks (CNN) using 5-fold cross-validation on the training dataset.
Assess the performance of GAM, GAN, and CNN on the validation dataset.


