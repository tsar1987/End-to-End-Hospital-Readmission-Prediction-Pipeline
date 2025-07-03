# src\train.py

"""
This script trains a Random Forest classifier to predict hospital readmission
for diabetic patients. The script performs the following steps:
1. Initializes a Spark Session.
2. Loads the diabetic patient data.
3. Preprocessing data includes clean data, impute data and creat new features. Using preprocessing.py
4. Create pipeline.
5  Trains and evaluate a weighted Random Forest model using AUC, Recall, Precision, and a Confusion Matrix.
6. Save model, metrics and Confusion Matrix
7. Stops the Spark Session.
"""
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = "C:\\hadoop" 
os.environ['PATH'] = f"{os.environ['HADOOP_HOME']}\\bin;{os.environ['PATH']}"

# --- Imports ---
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
import json

# Import other functions from src\preprocessing.py
from preprocessing import clean_data, impute_data, create_new_feature

# --- Load Data ---
def load_training_data(spark: SparkSession, data_path: str) -> DataFrame:
    """Loads the raw diabetic data from a CSV file."""
    print(f"Loading raw data from {data_path}...")
    df = spark.read.csv(data_path, header=True)
    print(f"Data loaded successfully. Initial rows: {df.count()}")
    return df

def create_pipeline(df: DataFrame) -> Pipeline:
    """
    Creates a feature engineering pipeline for the cleaned data.
    - Identifies categorical columns.
    - Applies StringIndexer and OneHotEncoder.
    - Assembles all features into a single vector.
    - Creates a full machine learning pipeline.
    """
    print("Starting create pipeline...")

    categorical_cols = [c for c, t in df.dtypes if t == 'string' and c != 'label' and c!= 'classWeight']
    numerical_cols = [c for c, t in df.dtypes if (t == 'int' or t == 'double') and c != 'label' and c!= 'classWeight']

    # Preprocessing Pipeline
    stages = []
    for col_name in categorical_cols:
        string_indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="keep")
        one_hot_encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[col_name + "_vec"])
        stages += [string_indexer, one_hot_encoder]
    
    assembler_inputs = [c + "_vec" for c in categorical_cols] + numerical_cols
    vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    stages += [vector_assembler]
    
    preprocessing_pipeline = Pipeline(stages=stages)

    # Define the machine learning model
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol='classWeight', seed=42)

    # Chain the feature pipeline and model into a full pipeline
    full_pipeline = Pipeline(stages=[preprocessing_pipeline, rf])
    print("Pipeline created successfully!")
    
    return full_pipeline

def train_evaluate_model(pipeline, trainData, testData):
    """
    Train and evaluate the model.
    """
    print("Starting train the model...")
    pipeline_model = pipeline.fit(trainData)
    print("Training complete!")

    # Evaluate model performance
    predictions = pipeline_model.transform(testData)

    print("Starting model evaluation...")
    # AUC
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    print(f"Area Under ROC (AUC) = {auc:.4f}")

    # Recall
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0)
    recall = evaluator_recall.evaluate(predictions)
    print(f"Recall (for Readmitted Class 1.0) = {recall:.2%}")

    # Precision
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0)
    precision = evaluator_precision.evaluate(predictions)
    print(f"Precision (for Readmitted Class 1.0) = {precision:.2%}")

    # Confusion Matrix
    preds_and_labels = predictions.select(['prediction', 'label']).toPandas()
    confusion_matrix_df = pd.crosstab(preds_and_labels['label'], preds_and_labels['prediction'])
    print("Model evaluation finished!")

    return pipeline_model, auc, recall, precision, confusion_matrix_df

def main():
    """Main function to run the training pipeline."""

    # --- Initialize Spark Session ---
    # Using 'local[*]' to use all available cores. The app name is updated.
    spark = SparkSession.builder.appName("HospitalReadmissionTrainer").getOrCreate()
    print("Spark session created successfully!")

    df_spark = load_training_data(spark, data_path='data/diabetic_data.csv')
    
    # Train test split
    (train_raw_df, test_raw_df) = df_spark.randomSplit([0.8, 0.2], seed=42)

    print(f"Data split into training ({train_raw_df.count()} rows) and test ({test_raw_df.count()} rows) sets.")

    # Preprocess Training Data
    print("Starting process train data...")
    train_cleaned = clean_data(train_raw_df)
    train_imputed = impute_data(train_cleaned)
    train_final = create_new_feature(train_imputed)
    print("Train data processed successfully!")
    
    # Preprocess Test Data
    print("Starting process test data...")
    test_cleaned = clean_data(test_raw_df)
    test_imputed = impute_data(test_cleaned)
    test_final = create_new_feature(test_imputed)
    print("Test data processed successfully!")
    
    # Create pipeline
    full_pipeline = create_pipeline(train_final)

    pipeline_model, auc, recall, precision, confusion_matrix_df = train_evaluate_model(full_pipeline, trainData=train_final, testData=test_final)

    # Print
    print(f"auc:{auc}, recall:{recall}, precision:{precision}")
    print("\nConfusion Matrix:")
    print(confusion_matrix_df)
    
    # Save model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "output", "spark_random_forest_model")
    pipeline_model.write().overwrite().save(model_path)
    print("Model saved successfully!")
    
    # Save metrics
    metrics = {"areaUnderROC": auc, "recall": recall, "precision": precision}
    with open(os.path.join(os.path.join(project_root, "output"), "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved successfully!")
    
    # Save confusion matrix
    with open(os.path.join(os.path.join(project_root, "output"), "confusion_matrix.txt"), 'w') as f:
        f.write(str(confusion_matrix_df))
    print(f"Confusion matrix saved successfully!")

    # --- Stop Spark Session ---
    print("Stopping Spark session!")
    spark.stop()

# ==============================================================================
# This is the standard entry point for a Python script.
# ==============================================================================
if __name__ == "__main__":
    main()

