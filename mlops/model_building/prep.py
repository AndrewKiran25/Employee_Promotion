# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
hf_token = os.getenv("token1")

if not hf_token:
    raise ValueError("token1 is missing!")

api = HfApi(token=hf_token)

DATASET_PATH = "hf://datasets/Andrew2505/Employee-Promotion/employee_promotion_final.csv"
promotion_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'is_promoted'

# List of numerical features in the dataset
numeric_features = [
    'no_of_trainings',
    'age',
    'previous_year_rating',
    'length_of_service',
    'awards_won',
    'avg_training_score',
]

# List of categorical features in the dataset
categorical_features = [
    'department',
    'region',
    'gender',
    'recruitment_channel',       # Country where the customer resides
]

# Separate education
ordinal_feature = ['education']

# Define predictor matrix (X) using selected numeric and categorical features
X = promotion_dataset[numeric_features + categorical_features + ordinal_feature]

# Define target variable
y = promotion_dataset[target]

# Split the dataset into training, validation and test sets
# Step 1: Train + Temp (80%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# then we split the temporary set into train and validation
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xval.to_csv("Xval.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
yval.to_csv("yval.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xval.csv","Xtest.csv","ytrain.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Andrew2505/Employee-Promotion",
        repo_type="dataset",
    )
