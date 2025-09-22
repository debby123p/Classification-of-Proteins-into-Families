# Classification-of-Proteins-into-Families-using-Machine-Learning

## Overview
This project aims to classify proteins into their respective family types based on a set of structural and experimental features. The primary challenge addressed is the inherent complexity and class imbalance found in biological datasets. Various machine learning models are trained, evaluated, and tuned to find the most effective classifier for this task.

## Project Workflow
The project follows a systematic machine learning pipeline:

### Data Cleaning & Preprocessing:

- Initial loading of the protein training data and class labels.

- Correction of structural misalignments and errors within the raw CSV data.

- Dropping of non-contributory features such as structureId, publicationYear, and pdbxDetails to reduce noise.

- Imputation of missing values using the most_frequent strategy.

- Consolidation of categories within the crystallisation method feature to reduce dimensionality.

### Data Balancing:

- The dataset is highly imbalanced. To handle this, classes with fewer than 900 samples were filtered out.

- The Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data to create synthetic samples for the minority classes, resulting in a balanced training set.

### Model Training & Evaluation:

- The preprocessed data was split into an 80% training set and a 20% testing set.

- Several classification models were trained and evaluated:

         - K-Nearest Neighbours (KNN)

         - Decision Tree Classifier (DTC)

         - Random Forest Classifier (RFC)

Model performance was measured using accuracy, precision, recall, and F1-score.

### Hyperparameter Tuning:

GridSearchCV was used to systematically search for the optimal hyperparameters for the evaluated models to improve their performance.

## Results and Discussion
After training and evaluation, the models showed varied performance. The Random Forest Classifier achieved the highest accuracy and F1-score, making it the most balanced and effective model for this particular task.

### Performance Summary
The table below summarizes the performance of the models on the test set:



### General Observations

- Variation in Performance: The results highlight a significant variation in performance across the different models, with each responding differently to the dataset's complexities.

- Challenges in Classification: The accuracy scores suggest inherent challenges in classifying complex protein data, possibly due to its high dimensionality and intricate patterns.

- Conclusion: The Random Forest Classifier (RFC) was the best-performing model for this task based on accuracy and F1-score. Further improvements could potentially be achieved through more advanced feature engineering or by exploring more complex models.

