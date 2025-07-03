# src/preprocessing.py

"""
This script trains a Random Forest classifier to predict hospital readmission
for diabetic patients. The script performs the following steps:
1. Performs data cleaning by removing nulls and dropping high-missingness columns.
2. Impute data.
3. Engineers new features, including diagnosis groups, service utilization ratios,
    comorbidity counts, and medication change scores.
"""

# Imports
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, mean, substring

def clean_data(df: DataFrame) -> DataFrame:
    """
    Performs initial data cleaning on the raw DataFrame.
    - Replaces '?' with null.
    - Drops unnecessary identifier columns.
    - Maps the target variable 'readmitted' to a binary 0/1 format.
    """
    print("Starting data cleaning...")

    # 1. Replace '?' with null across all columns
    # A more robust way than listing columns manually
    for column in df.columns:
        df = df.withColumn(column, when(col(column) == '?', None).otherwise(col(column)))

    # 2. Drop columns that are identifiers or have too many missing values
    # 'weight' is often dropped in this dataset due to >90% missing values.
    cols_to_drop = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]
    existing_cols = df.columns
    cols_that_exist_and_should_be_dropped = [col for col in cols_to_drop if col in existing_cols]
    if cols_that_exist_and_should_be_dropped:
        df = df.drop(*cols_that_exist_and_should_be_dropped)
        print(f"Dropped columns: {cols_that_exist_and_should_be_dropped}")

    # 3. Transform target variable 'readmitted' into 'label'
    # We want to predict early readmission (<30 days)
    # '<30' becomes 1 (positive class), '>30' and 'NO' become 0 (negative class)
    if 'readmitted' in df.columns:
        df = df.withColumn("label", when(col("readmitted") == "NO", 0.0).otherwise(1.0)).drop("readmitted")
        print("Transformed 'readmitted' target variable to binary 'label'.")
    
    # 4. Ensure numeric columns are cast correctly
    # This is important after replacing '?' with nulls
    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses']
    
    for col_name in numeric_features:
        df = df.withColumn(col_name, col(col_name).cast('integer'))


    print("Data cleaning complete!")
    return df

def impute_data(df: DataFrame) -> DataFrame:
    """
    Performs data imputation on the DataFrame.
    - Fillna with mean for numeric columns
    - Fillna with mode for categorical columns
    """
    print("Starting data imputation...")
    # Define features
    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                        'num_medications', 'number_outpatient', 'number_emergency', 
                        'number_inpatient', 'number_diagnoses']
    # 'diag_1', 'diag_2', 'diag_3' are handled in next step, not here.
    categorical_features = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 
                            'admission_source_id', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 
                            'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 
                            'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                            'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 
                            'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
                            'metformin-pioglitazone', 'change', 'diabetesMed']
    # Fillna for numeric columns
    mean_row = df.select([mean(col(feature)).alias(feature) for feature in numeric_features]).collect()[0].asDict()
    df = df.fillna(mean_row)
    # Fillna for categorical columns
    for feature in categorical_features:
        mode_val = (
            df.groupBy(feature).count().orderBy('count', ascending=False).first()[0])
        
        df = df.fillna({feature: mode_val})

    print("Data imputation finished!")
    return df
    
def create_new_feature(df: DataFrame) -> DataFrame:
    """
    Create some new features
    - Extract from 'diag_1', 'diag_2', 'diag_3'
    - Create 'comorbidity_count'
    - Create 'labs_per_day', 'meds_per_day', 'procs_per_day'
    - Create 'med_change_score'
    - Creat 'med_change_score'
    """
    print("Starting creating new features...")
    # Extract information from 'diag_1', 'diag_2', 'diag_3' and fillna
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for diag_col in diag_cols:
        group_col_name = f"{diag_col}_group"
        df = df.withColumn(
            group_col_name,
            when(col(diag_col).rlike("^V"), "V_Code")
            .when(col(diag_col).rlike("^E"), "E_Code")
            .when(col(diag_col).isNotNull(), substring(col(diag_col), 1, 3))
            .otherwise(None)
        )

    fill_dict = {f"{c}_group": "UNK" for c in diag_cols}
    df = df.fillna(fill_dict)
    df = df.drop(*diag_cols)

    # Create feature "comorbidity_count"
    major_disease_groups = ['250', '428', '414', '401', '585', '486', '493', '272']
    df = df.withColumn('comorbidity_count', col('time_in_hospital') * 0)

    for group in major_disease_groups:
        df = df.withColumn(
            'comorbidity_count',
            col('comorbidity_count') + when((col('diag_1_group') == group) | (col('diag_2_group') == group) | (col('diag_3_group') == group), 1).otherwise(0)
        )

    # Create features 'labs_per_day', 'meds_per_day', 'procs_per_day'
    df = df.withColumn('labs_per_day', col('num_lab_procedures') / col('time_in_hospital')) \
            .withColumn('meds_per_day', col('num_medications') / col('time_in_hospital')) \
            .withColumn('procs_per_day', col('num_procedures') / col('time_in_hospital'))
    
    # Creat feature 'med_change_score'
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
                'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
                'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    df = df.withColumn('med_change_score', col('time_in_hospital') * 0)
    for med in med_cols:
        df = df.withColumn('med_change_score', col('med_change_score') + when((col(med) == 'Up') | (col(med) == 'Down'), 1).otherwise(0))

    # Create 'classWeight'
    if "label" in df.columns:
        num_positives = df.filter(col('label') == 1.0).count()
        num_negatives = df.filter(col('label') == 0.0).count()
        weight_for_negatives = num_positives / (num_positives + num_negatives)
        weight_for_positives = num_negatives / (num_positives + num_negatives)
        df = df.withColumn('classWeight', when(col('label') == 0.0, weight_for_negatives).otherwise(weight_for_positives))

    print("New features created successfully!")
    return df
