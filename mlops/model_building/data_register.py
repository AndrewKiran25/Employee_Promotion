from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Andrew2505/Employee-Promotion"
repo_type = "dataset"

hf_token = os.getenv("TOKEN1")

if not hf_token:
    raise ValueError("TOKEN1 is missing!")

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False,token=hf_token)
    print(f"Dataset '{repo_id}' created.")

api.upload_folder(
    folder_path="mlops/data",
    repo_id=repo_id,
    repo_type=repo_type
)
