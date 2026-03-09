# Employee Attrition Prediction with Neural Networks

Deep learning model to predict employee attrition and department fit using a branched neural network architecture. Built with TensorFlow/Keras on HR analytics data to help organizations identify flight risk and optimize talent placement.

---

## Key Results

- **Branched neural network** with shared layers + two output heads: attrition prediction and department classification
- Attrition output: binary classification (likely to leave vs. stay)
- Department output: multi-class classification across business units
- Feature engineering on HR metrics including satisfaction scores, tenure, overtime, and performance ratings

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

- **Architecture:** Multi-output neural network with shared feature extraction layers
- **Preprocessing:** StandardScaler normalization, OneHotEncoding for categorical features
- **Evaluation:** Accuracy metrics on both attrition and department prediction tasks

---

## Project Structure

```
├── Attrition_Final.ipynb    # Main model notebook
└── README.md
```

---

## Methodology

1. Loaded and preprocessed HR dataset (satisfaction scores, salary, tenure, overtime, etc.)
2. Encoded categorical features and scaled numerical inputs
3. Built branched Keras model — shared dense layers split into two output heads
4. Trained and evaluated on attrition (binary) and department fit (multi-class)
5. Tuned activation functions and layer depth to optimize dual-task performance
