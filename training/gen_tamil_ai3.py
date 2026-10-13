"""3 LONG Tamil AI utterances (~10s each -> 3 segs each) for balance."""
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

OUT = r"C:\Users\Kishore GK\Downloads"
TEXTS = [
    "Vanakkam. Indha thittam seyal nunarivu kuralai kandupidikuvadhu. Nambikkaiyodu pesungal, ungal kural ungal adayalam. Edhirkaalam varum bodhu, idhu migavum mukkiyam.",
    "Tamil mozhi miga pazhamaiyaana inimaiyaana mozhi. Indha kural manidhanudaiyadha alladhu endhiram coatruvadha. Sothanai seidhu paarppom, unmai velivarum.",
    "Naan oru endhira kural, aanaal manidhan pola pesa muyarchikkiren. Indha sothanai vetri petraal, payanam thodarum. Nandri, meendum sandhippom.",
]
print('loading facebook/mms-tts-tam...', flush=True)
model = VitsModel.from_pretrained('facebook/mms-tts-tam')
tok = AutoTokenizer.from_pretrained('facebook/mms-tts-tam')
for i, text in enumerate(TEXTS, 7):
    with torch.no_grad():
        wav = model(**tok(text, return_tensors='pt')).waveform[0].numpy()
    sf.write(f'{OUT}\\tamil_ai_c{i}.wav', wav, 16000)
    print(f'saved tamil_ai_c{i}.wav ({len(wav)/16000:.1f}s)', flush=True)
print('TAMIL AI LONG DONE')
