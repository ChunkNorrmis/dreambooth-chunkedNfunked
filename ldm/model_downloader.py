import os, sys, gdown, requests, re, urllib
from huggingface_hub.file_download import hf_hub_download
import hf_xet, hf_transfer



def get_model(url):
    if url.startswith('http://'):
        url = url.replace('http://', 'https://')
    if url.startswith(('https://huggingface.co/', 'https://www.huggingface.co/')):
        return from_huggingface_hub(url)
    elif url.startswith(('https://drive.google.com/', 'https://www.drive.google.com/')):
        return from_google_drive(url)
    else:
        return from_generic(url)


def from_generic(url):
    response = requests.head(url, allow_redirects=True)
    header = response.headers.get('content-disposition')
    results = re.search(r'filename="?([^";]+)"?', header)
    ckpt_file = match.group(1)
    model_path = os.path.join(sys.path[0], ckpt_file)
    if not os.path.exists(model_path):
        print(f"Downloading '{ckpt_file}'...")
        urllib.response.urlretrieve(url, model_path)
    return os.path.relpath(model_path)


def from_huggingface_hub(url):
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    ckpt_file = os.path.basename(url)
    model_path = os.path.join(sys.path[0], ckpt_file)
    if not os.path.exists(model_path):
        print(f"Downloading '{ckpt_file}'...")
        hf_hub_download(repo_id, ckpt_file, local_dir=sys.path[0])
    return os.path.relpath(model_path)


def from_google_drive(url):
    def on_progress(bytes_so_far: int, bytes_total: int | None) -> None:
        if bytes_total is not None:
            print(f"\r{bytes_so_far / bytes_total * 100:.1f}%", end="")

    ckpt_file = gdown.download(url=url, skip_download=True)[-1]
    model_path = os.path.join(sys.path[0], ckpt_file)
    if not os.path.exists(model_path):
        print(f"Downloading '{ckpt_file}'...")
        gdown.download(url=url, output=model_path, quiet=True, progress=on_progress)
    return os.path.relpath(model_path)


