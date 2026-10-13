"""Search Kaggle for diverse audio deepfake datasets."""
import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_db9009dc7f361fbaf04b6500575791bb'
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print('Kaggle search: audio deepfake datasets\n')

queries = ['audio deepfake', 'asvspoof', 'wavefake', 'fake voice detection', 'speech spoofing']
seen = set()
for q in queries:
    try:
        ds = api.dataset_list(search=q, sort_by='votes')[:8]
        print(f'=== "{q}" ===')
        for d in ds:
            ref = d.ref
            if ref in seen:
                continue
            seen.add(ref)
            size = (getattr(d, 'totalBytes', None) or getattr(d, 'size', 0) or 0) / 1e6
            votes = getattr(d, 'voteCount', getattr(d, 'votes', 0)) or 0
            title = getattr(d, 'title', '') or ''
            print(f'  {ref:50s} | {size:8.0f} MB | votes {votes} | {title[:40]}')
    except Exception as e:
        print(f'  query "{q}" failed: {e}')
    print()
