"""Unit test: evidence engine on real + fake segments. Prints flags + timing."""
import sys, time, glob
sys.path.insert(0, r"C:\Users\Kishore GK\Desktop\VIDS\training")
from evidence import validate_audio, extract_evidence, flag_findings

BAL = r"C:\Users\Kishore GK\Desktop\VIDS\dataset\balanced"

for cls in ('real', 'fake'):
    print(f'===== {cls.upper()} (3 files) =====')
    for p in sorted(glob.glob(f'{BAL}/{cls}/*.wav'))[:3]:
        ok, reason = validate_audio(p)
        t0 = time.time()
        ev = extract_evidence(p)
        ms = round((time.time() - t0) * 1000)
        F, L = flag_findings(ev)
        print(f'[{p.split(chr(92))[-1]}] valid={ok} F0={ev["f0_mean"]}±{ev["f0_std"]} '
              f'voiced={ev["voiced_frac"]} HF={ev["hf_ratio"]} rms_std={ev["rms_std"]} ({ms}ms)')
        for f in F:
            print(f'   ({f["direction"]:5s}) [{f["tag"]}] {f["text"][:90]}')
        for lim in L:
            print(f'   (limit) {lim}')
    print()
