Breast Cancer Proteomic Analysis Project

Overview

This project analyzes proteomic and clinical data to predict Estrogen Receptor (ER) status in breast cancer patients using machine learning. It processes three datasets, applies feature engineering, trains four models, and visualizes results to identify key proteins and model performance. The project achieves high classification accuracy, with Random Forest scoring up to 95%.

Datasets

https://www.kaggle.com/datasets/piotrgrabo/breastcancerproteomes/code 

77_cancer_proteomes_CPTAC_itraq.csv: Proteomic data (12.4 MB)



clinical_data_breast_cancer.csv: Clinical data (18.6 KB)



PAM50_proteins.csv: PAM50 protein data (6.7 KB)

Requirements

Python 3.11



Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, upsetplot



Install dependencies:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn upsetplot

Usage





Upload Files: Run in Google Colab and upload the three CSV files.



Data Processing: Loads data, normalizes sample IDs, merges datasets, imputes missing values (median), and scales features using RobustScaler.



Feature Selection: Selects top 50 features using SelectKBest (ANOVA F-value).



Model Training: Trains and evaluates four models using 5-fold stratified cross-validation.



Visualization: Generates:





Bar plots of model performance (accuracy, precision, recall, F1-score).



Feature importance plot for top 20 proteins.



UpSet plot for co-occurrence of top protein expressions.

Machine Learning Models

The project uses four models, each configured for optimal performance on proteomic data:





XGBoost (Extreme Gradient Boosting)





Description: A gradient boosting algorithm that builds sequential decision trees, optimized for speed and accuracy with regularization to prevent overfitting.



Configuration:





n_estimators=300, max_depth=6, learning_rate=0.05



subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.1



eval_metric='logloss'



Performance: Accuracy 90%, strong precision/recall for both classes.



Why Used: Excels in tabular data, handles imbalance, and provides feature importance.



Random Forest





Description: An ensemble of decision trees using bootstrap sampling and random feature selection, robust to noise and overfitting.



Configuration:





n_estimators=300, max_depth=12, min_samples_split=3, min_samples_leaf=2



class_weight='balanced', max_features='sqrt'



Performance: Highest accuracy (95%), excellent precision/recall (0.92–0.96).



Why Used: Robust for high-dimensional data, interpretable feature importance.



Gradient Boosting





Description: Sequential tree-building to correct errors, similar to XGBoost but implemented via scikit-learn.



Configuration:





n_estimators=300, learning_rate=0.05, max_depth=5



min_samples_split=5, min_samples_leaf=2



Performance: Accuracy 85%, slightly lower than XGBoost.



Why Used: Captures complex patterns, complementary to XGBoost.



MLP (Multi-layer Perceptron)





Description: A neural network with multiple layers to learn non-linear relationships.



Configuration:





hidden_layer_sizes=(150, 75, 30), max_iter=1000, early_stopping=True



learning_rate='adaptive', alpha=0.001, batch_size=32



Performance: Accuracy 86%, competitive but below Random Forest.



Why Used: Explores non-linear patterns, alternative to tree-based models.

Mixture of Experts (MoE) Context

While not a formal MoE, the project uses multiple models as "experts" to leverage their strengths:





MoE Concept: MoE combines specialized models (experts) with a gating mechanism to select or weigh predictions based on input.



Project Approach: The four models act as independent experts, each handling the data differently (e.g., Random Forest for robustness, MLP for non-linearity). No gating mechanism is used; instead, models are compared directly.



Why Not MoE?: Simplicity and interpretability are prioritized, and Random Forest’s 95% accuracy reduces the need for a complex MoE setup.



Potential Extension: Add a gating network (e.g., neural network) to combine model predictions dynamically for improved performance.

Output


Console: Displays data shapes, missing values (before: 104,131; after: 0), classification reports, and metrics.



Visualizations:

Bar plots comparing model metrics (80–100% range).



Feature importance plot (top 20 proteins).



UpSet plot showing co-occurrence of top 10 protein expressions.



Performance: Random Forest leads (95% accuracy), followed by XGBoost (90%), MLP (86%), and Gradient Boosting (85%).
