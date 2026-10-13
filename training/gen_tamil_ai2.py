"""3 MORE Tamil AI utterances (different texts) for the Tamil fine-tune round."""
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

OUT = r"C:\Users\Kishore GK\Downloads"
TEXTS = [
    "Indha thittam seyal nunarivu kuralai kandupidikuvadhu.",
    "Nambikkaiyodu pesungal, ungal kural ungal adayalam.",
    "Edhirkaalam varum bodhu, idhu migavum mukkiyam.",
]
print('loading facebook/mms-tts-tam...', flush=True)
model = VitsModel.from_pretrained('facebook/mms-tts-tam')
tok = AutoTokenizer.from_pretrained('facebook/mms-tts-tam')
for i, text in enumerate(TEXTS, 4):
    with torch.no_grad():
        wav = model(**tok(text, return_tensors='pt')).waveform[0].numpy()
    sf.write(f'{OUT}\\tamil_ai_b{i}.wav', wav, 16000)
    print(f'saved tamil_ai_b{i}.wav ({len(wav)/16000:.1f}s)', flush=True)
print('TAMIL AI ROUND 2 DONE')
