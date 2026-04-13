# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# for handling imbalanced datasets
from imblearn.over_sampling import SMOTE

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/Andrew2505/Employee-Promotion/Xtrain.csv"
Xtest_path = "hf://datasets/Andrew2505/Employee-Promotion/ytrain.csv"
ytrain_path = "hf://datasets/Andrew2505/Employee-Promotion/Xtest.csv"
ytest_path = "hf://datasets/<Andrew2505/Employee-Promotion/Xtest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Numerical features
numeric_features = [
    'no_of_trainings',
    'age',
    'previous_year_rating',
    'length_of_service',
    'awards_won',
    'avg_training_score'
]

# Categorical features
categorical_features = [
    'department',
    'region',
    'gender',
    'recruitment_channel'
]

# Separate education
ordinal_feature = ['education']

# Define the preprocessing steps
# Step 1: Encoding (NO scaling here)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # keep as is
        ('ord', OrdinalEncoder(), ordinal_feature),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)

# Define hyperparameter grid
param_grid = {
    'model__n_estimators': [50, 100, 300],    # number of tree to build
    'model__learning_rate': [0.01,0.1,0.2,0.05],    # learning rate
    'model__gamma':[0,1,3,5],
    'model__subsample':[0.7,0.8,0.9,1],
    'model__max_depth': [3, 5, 7],
    'model__min_child_weight': [1, 3, 5],
    'model__colsample_bytree': [0.6, 0.8, 1]
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor), #encoding
    ('smote', SMOTE(sampling_strategy=0.7, k_neighbors=5, random_state=42)),
    ('model', xgb_model)
])

with mlflow.start_run():

    #Calling RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator = model_pipeline,
        param_distributions=param_grid,
        n_jobs = -1,
        n_iter=20,
        scoring='f1',
        cv=kfold,
        verbose=1,
        random_state=42)

    random_search.fit(Xtrain, ytrain)

    results = random_search.cv_results_

    # FIXED LOOP
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    mlflow.log_params(random_search.best_params_)

    best_model = random_search.best_estimator_

    # Get validation probabilities FIRST
    y_pred_val_proba = best_model.predict_proba(Xval)[:, 1]

    # Find best threshold using validation set
    best_threshold = 0
    best_f1 = 0

    thresholds = [
    0.1, 0.15, 0.2, 0.25, 0.3,
    0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7,
    0.75, 0.8, 0.85, 0.9]

    for t in thresholds:
        y_val_temp = (y_pred_val_proba >= t).astype(int)
        f1 = classification_report(yval, y_val_temp, output_dict=True)['1']['f1-score']

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    classification_threshold = best_threshold
    print("Best threshold:", classification_threshold)

    # Train predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    # validation predictions
    y_pred_val_proba = best_model.predict_proba(Xval)[:, 1]
    y_pred_val = (y_pred_val_proba >= classification_threshold).astype(int)

    # Test predictions
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    val_report = classification_report(yval, y_pred_val, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)


    # Log metrics to MLflow
    mlflow.log_metrics({
    # Train metrics
    "train_accuracy": train_report['accuracy'],
    "train_precision": train_report['1']['precision'],
    "train_recall": train_report['1']['recall'],
    "train_f1-score": train_report['1']['f1-score'],

    # Validation metrics
    "val_accuracy": val_report['accuracy'],
    "val_precision": val_report['1']['precision'],
    "val_recall": val_report['1']['recall'],
    "val_f1-score": val_report['1']['f1-score'],

    # Test metrics
    "test_accuracy": test_report['accuracy'],
    "test_precision": test_report['1']['precision'],
    "test_recall": test_report['1']['recall'],
    "test_f1-score": test_report['1']['f1-score']
})

    # Save the model locally
    model_path = "best_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Andrew2505/Employee_Promotion"
    repo_type = "model"

  hf_token = os.getenv("token1")

  if not hf_token:
    raise ValueError("token1 is missing!")

  api = HfApi(token=hf_token)

  # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_model_v1.joblib",
        path_in_repo="best_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
