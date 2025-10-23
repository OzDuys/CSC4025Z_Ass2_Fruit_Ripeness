# CSC4025Z (2025): Artificial Intelligence

## Assignment 2: Application of Neural Networks

**Lecturer:** Dr. Jan Buys — [jbuys@cs.uct.ac.za](mailto:jbuys@cs.uct.ac.za)
**TA:** Claytone Sikasote — [SKSCLA001@myuct.ac.za](mailto:SKSCLA001@myuct.ac.za)

**Deadline:** Monday, 27 October 2025 at 23:59
**Presentations:** Monday, 27 October 2025

---

## Overview

You should work in groups of **2 to 3 students**.
For any questions related to the project, please email **both** the lecturer and the TA.
Contact them early for any clarifications.

The goal of this project is to **develop a neural network-based AI system** and evaluate its performance.

You should implement the system using a **numerical computation or neural network library** — **PyTorch** is recommended, though **TensorFlow**, **Keras**, or other libraries are acceptable.

> ⚠️ You may use existing code as a starting point **with proper acknowledgment**, but you must **train, validate, and test** the model yourself.
> **Do not** use or fine-tune pretrained neural networks for this assignment.

---

## Recommended Platforms

You may train and run your models using **Google Colab** or **Kaggle Notebooks**.
Neural networks typically train more efficiently on **GPUs**, which are available through Colab.

> Note: Free Colab accounts have **time limits** on GPU usage — do not leave training to the last minute.

* [PyTorch](https://pytorch.org/)
* [Google Colab](https://colab.research.google.com/)
* [Kaggle Code](https://www.kaggle.com/code)
* [Kaggle Datasets](https://www.kaggle.com/datasets)

---

## Project Steps

### 1. Choose a Dataset

Use any **publicly available dataset** suitable for **supervised classification**.
Good sources include [Kaggle](https://www.kaggle.com/datasets).

---

### 2. Formulate the Problem

Define a **multi-class classification task** that predicts an attribute in your dataset.

* Aggregate discrete or continuous values into classes if appropriate.
* Ensure **more than two output classes**.
* Clearly define **inputs and outputs**.
* Split the dataset into **training**, **validation**, and **test** sets.
* Select an appropriate **evaluation metric**.
* Reflect on any **ethical implications** of the AI system you build.

---

### 3. Pick a Baseline

Choose the simplest reasonable model as a baseline. Examples include:

* Count-based probability estimates
* **Naïve Bayes**
* **K-Nearest Neighbours (KNN)**

---

### 4. Develop the Model

Design and implement your neural network:

* Select an appropriate **architecture** and **input representation**.
* Train and tune the model.
* Explore different **hyperparameters**, **features**, or **architectures**.
* Conduct a **final evaluation**.

---

### 5. Analyse Model Performance

Your grade does **not depend on high accuracy**, but rather on your **depth of analysis**.

Discuss:

* Model design choices
* Optimization process
* Evaluation methodology
* Insights into errors or limitations

---

## Deliverables

### 1. Project Report

Suggested length: **≈5 pages** (guideline only)

Include:

* Problem description
* Baseline and model design choices
* Experimental setup
* Results (validation and test sets)
* Analysis of model performance

---

### 2. Code and Data

Submit:

* Your full project code
* Any **external code** used (clearly acknowledged)
* Test data (or sample if >10 MB)
* Model’s **final predictions** on the test set
* Instructions or a **reproduction script**

---

### 3. Final Presentation

Each group will give a **5-minute presentation** covering:

* Problem formulation
* Model architecture
* Evaluation and analysis of performance

> ⏱️ The 5-minute time limit will be **strictly enforced**.

---

## Submission

The **group leader** must submit a **single ZIP file** on **Amathuba**, containing:

* Report
* Code
* Data
* Presentation slides

Name the file with the **student numbers** of all group members.
Late submissions will incur a **10% penalty per day**.

---

## Marking Rubric (Total: 30 Marks)

| **Criteria**                                                          | **Marks** |
| --------------------------------------------------------------------- | --------- |
| Problem formulation (classification task, usefulness, ethics)         | 3         |
| Baseline (appropriate baseline included)                              | 3         |
| Model design (architecture and features)                              | 3         |
| Model validation (hyperparameter tuning and optimization)             | 3         |
| Evaluation (metrics, splits, and results)                             | 3         |
| Analysis of model performance                                         | 3         |
| Software (working code and appropriate tools)                         | 3         |
| Reproducibility (test results, parameters, reproduction instructions) | 3         |
| Overall report quality                                                | 3         |
| Final presentation quality                                            | 3         |
| **Total**                                                             | **30**    |