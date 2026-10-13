"""Download + unzip a Kaggle dataset. Token read from ~/.kaggle/access_token.

Usage: python kaggle_dl.py <owner/slug> <outdir>
"""
import os, sys, zipfile

os.environ['KAGGLE_API_TOKEN'] = open(
    os.path.join(os.path.expanduser('~'), '.kaggle', 'access_token')).read().strip()

from kaggle.api.kaggle_api_extended import KaggleApi

ref, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
api = KaggleApi()
api.authenticate()
print(f'Downloading {ref} -> {outdir}', flush=True)
api.dataset_download_files(ref, path=outdir, unzip=True, quiet=False)
print(f'DONE: {ref}', flush=True)
