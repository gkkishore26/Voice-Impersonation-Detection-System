"""Generate 4 fresh AI voices (MMS-TTS, 4 language checkpoints = 4 voices).

Guaranteed unseen: MMS-TTS is in NONE of our training sets.
~10s neutral speech each -> mp3 in Downloads.
"""
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

OUT = r"C:\Users\Kishore GK\Downloads"
JOBS = [
    ('facebook/mms-tts-eng', 'ai_test_1.mms_eng.mp3',
     "The quick brown fox jumps over the lazy dog. Voice cloning technology keeps improving every single year."),
    ('facebook/mms-tts-fra', 'ai_test_2.mms_fra.mp3',
     "Bonjour, ceci est un test de voix artificielle. La technologie avance tres rapidement aujourd hui."),
    ('facebook/mms-tts-deu', 'ai_test_3.mms_deu.mp3',
     "Guten Tag, dies ist eine kuenstliche Stimme. Die Erkennung solcher Stimmen ist eine spannende Aufgabe."),
    ('facebook/mms-tts-spa', 'ai_test_4.mms_spa.mp3',
     "Hola, esta es una voz generada por inteligencia artificial para probar el detector."),
]

for ckpt, fn, text in JOBS:
    print(f'loading {ckpt}...', flush=True)
    model = VitsModel.from_pretrained(ckpt)
    tok = AutoTokenizer.from_pretrained(ckpt)
    with torch.no_grad():
        wav = model(**tok(text, return_tensors='pt')).waveform[0].numpy()
    path = f'{OUT}\\{fn}.wav'
    sf.write(path, wav, 16000)
    print(f'saved {fn}.wav ({len(wav)/16000:.1f}s)', flush=True)
print('ALL 4 AI VOICES GENERATED')
