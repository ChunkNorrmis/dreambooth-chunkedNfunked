import os, sys
from huggingface_hub.file_download import hf_hub_download
import hf_xet, hf_transfer


def get_model_from_hf(repo_url):
    repo_id = f"{repo_url.split('/')[3]}/{repo_url.split('/')[4]}"
    ckpt_file = os.path.basename(repo_url)
    model_path = os.path.join(sys.path[0], ckpt_file)
    if not os.path.exists(model_path):
        print(f"Downloading '{ckpt_file}'")
        hf_hub_download(repo_id, ckpt_file, local_dir=sys.path[0])
    return os.path.relpath(model_path)

	
