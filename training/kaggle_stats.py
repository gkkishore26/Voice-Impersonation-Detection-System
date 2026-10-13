"""Get dataset object attrs + full file count/size with pagination."""
import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_db9009dc7f361fbaf04b6500575791bb'
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# 1) what attributes does ApiDataset have?
ds = api.dataset_list(search='in-the-wild-audio-deepfake')[0]
print('ApiDataset attrs:', [a for a in dir(ds) if not a.startswith('_')])
print()

def full_stats(ref):
    """page through all files"""
    total_bytes = 0
    count = 0
    page_token = None
    while True:
        resp = api.dataset_list_files(ref, page_token=page_token) \
            if page_token else api.dataset_list_files(ref)
        files = resp.files if hasattr(resp, 'files') else resp
        for f in files:
            total_bytes += (getattr(f, 'totalBytes', 0) or 0)
            count += 1
        page_token = getattr(resp, 'next_page_token', None)
        if not page_token or not files:
            break
    return count, total_bytes

for ref in [
    'jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech',
    'muhammadali03/wavefake-audio-16k',
    'abdallamohamed312/in-the-wild-audio-deepfake',
]:
    try:
        n, b = full_stats(ref)
        print(f'{ref}\n   -> {n} files, {b/1e9:.2f} GB\n')
    except Exception as e:
        print(f'{ref} FAILED: {e}\n')
