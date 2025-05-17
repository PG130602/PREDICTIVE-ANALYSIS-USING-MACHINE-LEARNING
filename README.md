# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : Pavan Vadiraj Gokak

*INTERN ID* : CT08WT194

*DOMAIN* : Data Analytics

*DURATION* : 8 WEEKS

*MENTOR* : Neela Santhosh Kumar 

# Loan Default Prediction using Machine Learning

This project presents a machine learning-based solution to predict whether a loan applicant is likely to default. The dataset includes various financial and demographic features, and the primary aim is to develop and evaluate classification models that can identify high-risk customers. Such models are useful for financial institutions to reduce lending risk and improve decision-making.

# 📚 Project Overview

The main objective is to build a predictive system that can classify loans into "Default" or "No Default" categories. To accomplish this, the project involves key steps such as data cleaning, feature engineering, model training, and performance evaluation. Multiple machine learning models are used to identify the best performing one, and the final model is saved for future use.

# Tools & Technologies

1.Programming Language: Python 3.x
2.Data Handling: pandas, numpy
3.Visualization: matplotlib, seaborn
4.Machine Learning Models: scikit-learn, xgboost
5.Model Saving: joblib
6.Development Environment: Google Colab

# 📁 Dataset
The dataset file, Loan_default.csv, contains customer and loan attributes. The target column is Default, indicating whether a customer has defaulted on a loan. Before training, the dataset is analyzed and cleaned to remove inconsistencies and fill missing values.

# 🔁 Workflow
1. Importing Libraries & Loading Data
All required Python libraries are imported, including those for preprocessing, modeling, and visualization. The dataset Loan_default.csv is then uploaded and loaded using pandas.

2. Exploratory Data Analysis (EDA)
Initial insights are gained through functions like .head(), .info(), and .describe(). Visualizations such as count plots and correlation heatmaps are used to better understand feature distributions and relationships.

3. Data Cleaning
Missing values are handled using the SimpleImputer with a “most frequent” strategy. Non-numeric and unnecessary columns (like IDs) are dropped. Categorical values are encoded using LabelEncoder.

4. Feature Scaling
Numerical features are standardized using StandardScaler to ensure consistency across models. This step helps improve model performance, especially for algorithms sensitive to feature magnitude.

5. Train-Test Split
The dataset is split into training and testing subsets in an 80-20 ratio. This ensures that model evaluation is done on unseen data.

6. Model Training
Two classification models are trained:

Logistic Regression

Random Forest Classifier

Each model is fitted on the training data and tested on the testing set. Accuracy and classification reports are generated.

7. Performance Evaluation
ROC curves are plotted for both models to compare true positive and false positive rates. The Area Under Curve (AUC) metric is used to rank performance. Confusion matrices are also generated to analyze prediction outcomes.

8. Cross-Validation
A 5-fold cross-validation technique is applied to evaluate the generalization performance of the model. Average accuracy scores across folds are calculated for consistency.

9. Saving the Model
The best-performing model (Random Forest) and the fitted scaler are saved using joblib, allowing future reuse without retraining.

# 📤 Output Summary
The main output of this project is the loan_default_model.pkl file, which contains the trained Random Forest Classifier saved using joblib. This model can be reused for predicting loan default outcomes on new data without needing to retrain, making it suitable for deployment in applications such as risk assessment tools or financial decision-support systems.

# 👤 Author
This project was developed by Pavan Gokak as part of a task for CODTECH IT SOLUTIONS, demonstrating the practical application of machine learning in the financial domain. It highlights how predictive modeling can support risk management by identifying potential loan defaulters. The project showcases a complete ML workflow—from data preprocessing and model training to evaluation and deployment—emphasizing the real-world value of data-driven decision-making in banking and credit analysis.

![Image](https://github.com/user-attachments/assets/3ca848b9-b8c3-44d8-b7cd-babd9aac267e)
![Image](https://github.com/user-attachments/assets/ebe8e18a-3773-4914-9f8e-5a3774508134)
![Image](https://github.com/user-attachments/assets/a484b88b-dddd-4292-8bbf-90ced5519c50)
![Image](https://github.com/user-attachments/assets/59f533db-1903-4424-a694-a8ffa2c2c6bc)
![Image](https://github.com/user-attachments/assets/36874235-951f-41e5-9e0e-3b3672ba6b98)
![Image](https://github.com/user-attachments/assets/8aa0c608-3214-4981-93dd-71cf4aba5748)
![Image](https://github.com/user-attachments/assets/10f4b24a-af4e-4954-92db-da60a2241044)
![Image](https://github.com/user-attachments/assets/fb7dee58-5ea7-4f19-8217-da5f919943c7)
![Image](https://github.com/user-attachments/assets/b304e35d-ad6f-4b64-9ef6-d35f32523591)
![Image](https://github.com/user-attachments/assets/688c8075-040d-4d0c-b14a-06c2da844d73)
![Image](https://github.com/user-attachments/assets/7fc9ba36-932a-48ae-a761-f6047bd6d690)
