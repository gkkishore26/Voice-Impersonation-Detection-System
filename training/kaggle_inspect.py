"""Inspect top candidate datasets: file list + sizes."""
import os, sys
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_db9009dc7f361fbaf04b6500575791bb'
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

CANDIDATES = [
    'abdallamohamed312/in-the-wild-audio-deepfake',
    'muhammadali03/wavefake-audio-16k',
    'jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech',
    'dinaahmed11/wavefake',
    'kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices',
]

for ref in CANDIDATES:
    print(f'=== {ref} ===')
    try:
        files = api.dataset_list_files(ref)
        fl = files.files if hasattr(files, 'files') else files
        total = 0
        for f in fl[:25]:
            sz = getattr(f, 'totalBytes', 0) or 0
            total += sz
            name = getattr(f, 'name', '?')
            print(f'   {name[:60]:62s} {sz/1e6:9.1f} MB')
        if len(fl) > 25:
            for f in fl[25:]:
                total += (getattr(f, 'totalBytes', 0) or 0)
            print(f'   ... +{len(fl)-25} more files')
        print(f'   TOTAL: {total/1e9:.2f} GB  ({len(fl)} files)\n')
    except Exception as e:
        print(f'   FAILED: {e}\n')
