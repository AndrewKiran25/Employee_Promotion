from huggingface_hub import HfApi
import os

hf_token = userdata.get("HF_TOKEN")
api = HfApi(token=hf_token)

api.upload_folder(
    folder_path="/content/mlops/deployment",     # the local folder containing your files
    repo_id="Andrew2505/Employee-Promotion",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
