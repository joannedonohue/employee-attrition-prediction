# Employee Attrition Prediction

HR analytics model predicting employee attrition risk and department fit using a branched multi-output neural network.

---

## Overview

This project applies deep learning to HR analytics data to simultaneously solve two classification problems: predicting whether an employee will leave the company, and determining their best-fit department. A shared-layer branched neural network architecture handles both tasks within a single training pipeline.

---

## Problem Statement

Employee attrition is costly and disruptive. Traditional HR models treat retention and role-fit as separate problems. This project uses a multi-output network to jointly optimize both predictions, enabling HR teams to identify at-risk employees and recommend appropriate departmental re-assignments in one pass.

---

## Model Architecture

```
Input Features (HR metrics)
        │
   Shared Dense Layers
   (feature extraction)
        │
   ┌────┴────┐
   │         │
Output 1    Output 2
(Attrition) (Department)
Binary      Multi-class
```

- **Shared layers:** Extract common HR feature representations
- **Output Head 1:** Binary classification — will the employee stay or leave?
- **Output Head 2:** Multi-class classification — which department is the best fit?
- **Preprocessing:** StandardScaler for numerical features; OneHotEncoding for categorical variables

---

## Input Features

HR metrics analyzed include:
- Employee satisfaction scores
- Salary level
- Tenure duration
- Overtime status
- Performance ratings

---

## Tech Stack

| Component | Tool |
|---|---|
| Deep learning framework | TensorFlow / Keras |
| Data preprocessing | scikit-learn (StandardScaler, OneHotEncoder) |
| Data manipulation | pandas |
| Language | Python |

---

## Methodology

1. Load and explore the HR dataset
2. Encode categorical variables and scale numerical features
3. Construct branched Keras model with shared and task-specific output layers
4. Train simultaneously on both classification targets
5. Tune activation functions and layer depth for both output heads
6. Evaluate performance separately for attrition and department prediction

---

## Repository Structure

```
employee-attrition-prediction/
├── employee_attrition_prediction.ipynb   # Full model implementation
└── README.md
```

---

## Outcomes

- Designed and trained a multi-output neural network handling two distinct classification objectives within a shared architecture
- Demonstrated how shared feature extraction reduces training overhead compared to separate models
- Produced a reusable HR analytics pipeline covering preprocessing, model construction, training, and evaluation
- Provides a foundation for HR teams to identify high-risk employees while simultaneously assessing role fit

---

## Getting Started

```bash
pip install tensorflow scikit-learn pandas
jupyter notebook employee_attrition_prediction.ipynb
```
