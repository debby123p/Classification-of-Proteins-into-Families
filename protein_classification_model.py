"""
Protein Family Type Classification using Machine Learning

This script trains and evaluates several machine learning models to classify protein family types.
The workflow includes:
1. Loading and cleaning the raw protein data.
2. Preprocessing the data (imputation, encoding, feature selection).
3. Balancing the dataset using SMOTE.
4. Training and evaluating Random Forest, K-Nearest Neighbors, and Decision Tree classifiers.
5. Saving the performance metrics and confusion matrix plots to files.

To run this script:
python train_model.py --data_path /path/to/your/protein_trn_data.csv --labels_path /path/to/your/protein_trn_class_labels.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

def load_and_clean_data(data_path, labels_path):
    """
    Loads the protein training data and class labels, performs initial cleaning and merging.

    Args:
        data_path (str): The file path for the protein training data CSV.
        labels_path (str): The file path for the protein class labels CSV.

    Returns:
        pandas.DataFrame: A cleaned and merged DataFrame containing features and labels.
    """
    print("Step 1: Loading and cleaning data...")
    # Fix rows with shifted columns in the main data file
    data = []
    with open(data_path) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > 14:
                row = row[1:15]  # Shift columns to the left
            data.append(row)
    
    headers = data[0]
    protrn = pd.DataFrame(data[1:], columns=headers)

    # Load class labels
    plabel = pd.read_csv(labels_path, header=None)
    plabel.columns = ['Num', 'class']
    plabel = plabel.drop('Num', axis=1)

    # Join data and labels
    pr = protrn.join(plabel, how='left')

    # Consolidate categories in 'crystallizationMethod'
    pr["crystallizationMethod"].replace({
        "VAPOR DIFFUSION, HANGING DROP": "VAPOR DIFFUSION",
        "VAPOR DIFFUSION, SITTING DROP": "VAPOR DIFFUSION",
        "hanging drop": "VAPOR DIFFUSION",
        "VAPOR DIFFUSION,SITTING DROP,NANODROP": "VAPOR DIFFUSION",
        "VAPOR DIFFUSION, SITTING DROP, NANODROP": "VAPOR DIFFUSION",
        "MICROBATCH": "MICROBATCH",
        "BATCH MODE": "MICROBATCH",
        "batch": "MICROBATCH",
        "microbatch under oil": "MICROBATCH",
        "SMALL TUBES": "Miscellaneous",
        "LIQUID DIFFUSION": "Miscellaneous"
    }, inplace=True)

    print("Data loading and cleaning complete.")
    return pr


def preprocess_data(df):
    """
    Performs data preprocessing including feature dropping, imputation, class filtering, and encoding.

    Args:
        df (pandas.DataFrame): The cleaned DataFrame.

    Returns:
        pandas.DataFrame: The fully preprocessed and encoded DataFrame.
    """
    print("Step 2: Preprocessing data...")
    # Drop columns with high cardinality that are not useful for classification
    df_processed = df.drop(['publicationYear', 'structureId', 'pdbxDetails'], axis=1)

    # Impute missing values using the most frequent value in each column
    imputer = SimpleImputer(strategy='most_frequent')
    df_processed = pd.DataFrame(imputer.fit_transform(df_processed), columns=df_processed.columns)
    df_processed = df_processed.dropna(how='any')

    # Filter out classes with fewer than 900 samples to handle extreme imbalance
    counts = df_processed['class'].value_counts()
    class_data = np.asarray(counts[(counts > 900)].index)
    df_filtered = df_processed[df_processed['class'].isin(class_data)]

    # Encode all categorical features to numerical values
    labelencoder = LabelEncoder()
    for column in df_filtered.columns:
        df_filtered[column] = labelencoder.fit_transform(df_filtered[column])

    print("Data preprocessing complete.")
    return df_filtered


def train_and_evaluate(df, output_dir="results"):
    """
    Trains, evaluates, and saves results for multiple classifiers.

    Args:
        df (pandas.DataFrame): The preprocessed DataFrame.
        output_dir (str): Directory to save results and plots.

    Returns:
        dict: A dictionary containing the performance metrics for each model.
    """
    print("Step 3: Training and evaluating models...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Separate features and target variable
    X = df.drop('class', axis=1)
    y = df['class']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to handle class imbalance in the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'KNeighbors': KNeighborsClassifier(n_neighbors=11, weights='distance', metric='manhattan'),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")

        # Create and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(15, 15))
        disp.plot(ax=ax, cmap='viridis')
        plt.title(f'Confusion Matrix - {name}')
        plot_path = os.path.join(output_dir, f'confusion_matrix_{name.lower()}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Confusion matrix saved to {plot_path}")

    print("\nModel training and evaluation complete.")
    return results

def save_results_summary(results, output_dir="results"):
    """
    Saves the performance metrics of all models to a text file.

    Args:
        results (dict): A dictionary of model performance metrics.
        output_dir (str): Directory to save the results file.
    """
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Protein Classification Model Performance Summary\n")
        f.write("="*50 + "\n")
        for name, metrics in results.items():
            f.write(f"\nModel: {name}\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Weighted Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Weighted Recall: {metrics['recall']:.4f}\n")
            f.write(f"  Weighted F1-Score: {metrics['f1_score']:.4f}\n")
    print(f"\nResults summary saved to {summary_path}")


def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Protein Family Type Classification Script")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the protein training data CSV file.")
    parser.add_argument('--labels_path', type=str, required=True, help="Path to the protein class labels CSV file.")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save output files.")
    
    args = parser.parse_args()

    # --- Run Pipeline ---
    # 1. Load and clean data
    cleaned_df = load_and_clean_data(args.data_path, args.labels_path)
    
    # 2. Preprocess data
    preprocessed_df = preprocess_data(cleaned_df)
    
    # 3. Train models and evaluate
    final_results = train_and_evaluate(preprocessed_df, args.output_dir)
    
    # 4. Save a summary of results
    save_results_summary(final_results, args.output_dir)
    
    print("\nPipeline finished successfully!")


if __name__ == '__main__':
    main()
