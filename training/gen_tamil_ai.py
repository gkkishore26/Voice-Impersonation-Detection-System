"""Tamil test set: AI Tamil (MMS-TTS tam) + real Tamil (Common Voice ta).

AI: 3 utterances via facebook/mms-tts-tam -> Downloads/tamil_ai_*.wav
Real: 15 validated clips from Common Voice ta -> dataset/kaggle/tamil_real/
"""
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

OUT = r"C:\Users\Kishore GK\Downloads"
TEXTS = [
    "Vanakkam, idhu oru seyal nunarivu kural sothanai.",
    "Tamil mozhi miga pazhamaiyaana inimaiyaana mozhi.",
    "Indha kural manidhanudaiyadha alladhu endhiram coatruvadha?",
]
print('loading facebook/mms-tts-tam...', flush=True)
model = VitsModel.from_pretrained('facebook/mms-tts-tam')
tok = AutoTokenizer.from_pretrained('facebook/mms-tts-tam')
for i, text in enumerate(TEXTS, 1):
    with torch.no_grad():
        wav = model(**tok(text, return_tensors='pt')).waveform[0].numpy()
    sf.write(f'{OUT}\\tamil_ai_{i}.wav', wav, 16000)
    print(f'saved tamil_ai_{i}.wav ({len(wav)/16000:.1f}s)', flush=True)
print('TAMIL AI DONE')
