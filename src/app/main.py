# File: app/main.py

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SparkSession
from src.preprocessing import clean_data, impute_data, create_new_feature
# --- 1. SETUP: This part runs only ONCE when the server starts ---
app = FastAPI(title="Hospital Readmission Prediction API", version="1.0")

# Create a simple, local Spark session to use the model
spark = SparkSession.builder.appName("ReadmissionInference").master("local[*]").getOrCreate()

# Load our saved model from the 'output' folder
# This path is relative to your main project directory
MODEL_PATH = "output/spark_random_forest_model" 

model = None  # Just a simple variable assignment. This is fast and has no dependencies.

def get_model():
    global model
    if model is None:
        # This code only runs when the function is CALLED for the first time.
        print("Loading model for the first time...")
        model = PipelineModel.load(MODEL_PATH) 
    return model

# --- 2. INPUT DEFINITION: Define the data our API expects ---
class PatientData(BaseModel):
    race: str = Field(..., example="Caucasian")
    gender: str = Field(..., example="Female")
    age: str = Field(..., example="[70-80)")
    admission_type_id: str = Field(..., example="1")
    discharge_disposition_id: str = Field(..., example="1")
    admission_source_id: str = Field(..., example="7")
    time_in_hospital: str = Field(..., example='5')
    num_lab_procedures: str = Field(..., example='40')
    num_procedures: str = Field(..., example='1')
    num_medications: str = Field(..., example='15')
    number_outpatient: str = Field(..., example='0')
    number_emergency: str = Field(..., example='0')
    number_inpatient: str = Field(..., example='0')
    number_diagnoses: str = Field(..., example='8')
    diag_1: str = Field(..., example="428")
    diag_2: str = Field(..., example="411")
    diag_3: str = Field(..., example="250")
    max_glu_serum: str = Field(..., example="None")
    A1Cresult: str = Field(..., example=">7")
    metformin: str = Field(..., example="No")
    repaglinide: str = Field(..., example="No")
    nateglinide: str = Field(..., example="No")
    chlorpropamide: str = Field(..., example="No")
    glimepiride: str = Field(..., example="No")
    acetohexamide: str = Field(..., example="No")
    glipizide: str = Field(..., example="No")
    glyburide: str = Field(..., example="No")
    tolbutamide: str = Field(..., example="No")
    pioglitazone: str = Field(..., example="No")
    rosiglitazone: str = Field(..., example="No")
    acarbose: str = Field(..., example="No")
    miglitol: str = Field(..., example="No")
    troglitazone: str = Field(..., example="No")
    tolazamide: str = Field(..., example="No")
    examide: str = Field(..., example="No")
    citoglipton: str = Field(..., example="No")
    insulin: str = Field(..., example="Steady")
    glyburide_metformin: str = Field(..., alias="glyburide-metformin", example="No")
    glipizide_metformin: str = Field(..., alias="glipizide-metformin", example="No")
    glimepiride_pioglitazone: str = Field(..., alias="glimepiride-pioglitazone", example="No")
    metformin_rosiglitazone: str = Field(..., alias="metformin-rosiglitazone", example="No")
    metformin_pioglitazone: str = Field(..., alias="metformin-pioglitazone", example="No")
    change: str = Field(..., example="Ch")
    diabetesMed: str = Field(..., example="Yes")

# --- 3. ENDPOINT: The URL that will make predictions ---
@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.dict(by_alias=True)
    spark_df = spark.createDataFrame([input_dict])
    spark_df_cleaned = clean_data(spark_df)
    spark_df_imputed = impute_data(spark_df_cleaned)
    final_df = create_new_feature(spark_df_imputed)
    pipeline_model = get_model()
    prediction_df = pipeline_model.transform(final_df)
    
    prediction = prediction_df.select("prediction").first()[0]
    probability_vector = prediction_df.select("probability").first()[0]
    probability_of_readmission = float(probability_vector[1])
    
    return {
        "prediction_label": "Readmitted" if prediction == 1.0 else "Not Readmitted",
        "prediction_value": int(prediction),
        "probability_of_readmission": probability_of_readmission
    }