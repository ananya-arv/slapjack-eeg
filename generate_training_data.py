"""
NeuroJack — Unicorn Hybrid Black Synthetic EEG Training Data Generator
=======================================================================
Generates realistic SSVEP EEG sessions matched exactly to the
g.tec Unicorn Hybrid Black hardware specs.

Unicorn Hybrid Black specs:
  - 8 EEG channels: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 (in order, 0-indexed)
  - Sample rate: 250 Hz (NOT 256!)
  - Resolution: 24-bit
  - Bandwidth: 0.1-60 Hz (hardware filtered)
  - Reference/Ground: Left & Right mastoids
  - LSL stream: UnicornLSL.exe streams 17 ch (EEG1-8, Accel x3, Gyro x3, Battery, Counter, Validation)

SSVEP channel strategy:
  Best:      Pz(4), PO7(5), Oz(6), PO8(7) — parieto-occipital, strongest SSVEP
  Support:   Cz(2), C4(3) — weaker SSVEP via volume conduction
  Reference: Fz(0), C3(1) — frontal, used as artifact reference
"""

import numpy as np
import csv
import os
import json
from datetime import datetime

# ── Unicorn Hardware Constants ───────────────────────────────────────────────
FS          = 250          # Unicorn exact sample rate
EPOCH_SEC   = 1.0
N_SAMPLES   = int(FS * EPOCH_SEC)   # = 250

CHANNELS        = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
N_EEG_CHANNELS  = 8
SSVEP_CHANNELS  = [4, 5, 6, 7]   # Pz, PO7, Oz, PO8 (0-indexed)
SUPPORT_CHANNELS = [2, 3]         # Cz, C4

FREQ_BASELINE = 2.0
FREQ_SLAP     = 10.0

EEG_AMP_UV  = 50.0
ALPHA_AMP   = 15.0


def pink_noise(n, alpha=1.0):
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-10
    power = freqs ** (-alpha / 2.0)
    spectrum = power * (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs)))
    sig = np.fft.irfft(spectrum, n=n)
    return sig / (np.std(sig) + 1e-10)


def simulate_unicorn_epoch(label, snr_db=None):
    """Simulate one Unicorn epoch. Returns (data [250x8], actual_snr_db)."""
    t    = np.linspace(0, EPOCH_SEC, N_SAMPLES, endpoint=False)
    data = np.zeros((N_SAMPLES, N_EEG_CHANNELS))

    for ch in range(N_EEG_CHANNELS):
        noise_amp = np.random.uniform(0.8, 1.2) * EEG_AMP_UV * 0.3
        bg        = noise_amp * pink_noise(N_SAMPLES)
        aw        = 1.1 if ch in SSVEP_CHANNELS else 0.7 if ch in SUPPORT_CHANNELS else 0.3
        alpha     = aw * ALPHA_AMP * np.sin(2*np.pi*np.random.uniform(9.5,11.5)*t + np.random.uniform(0,2*np.pi))
        drift     = np.random.uniform(0, EEG_AMP_UV*0.15) * np.sin(2*np.pi*np.random.uniform(0.2,0.8)*t + np.random.uniform(0,2*np.pi))
        line      = np.random.uniform(0, EEG_AMP_UV*0.04) * np.sin(2*np.pi*50*t + np.random.uniform(0,2*np.pi))
        data[:, ch] = bg + alpha + drift + line

    if snr_db is None:
        snr_db = np.random.uniform(6, 20)

    snr_lin    = 10 ** (snr_db / 20.0)
    bg_rms     = max(np.std(data[:, 6]), 1.0)   # Oz noise reference
    signal_amp = snr_lin * bg_rms

    if label == 'baseline':
        target_hz  = FREQ_BASELINE
        harmonic   = FREQ_BASELINE * 2
        ch_weights = {4: 0.65, 5: 0.90, 6: 1.00, 7: 0.90}  # Pz, PO7, Oz, PO8
        sup_w      = 0.20
    else:
        target_hz  = FREQ_SLAP
        harmonic   = FREQ_SLAP * 2
        ch_weights = {4: 0.70, 5: 0.92, 6: 1.00, 7: 0.92}
        sup_w      = 0.25

    for ch, weight in ch_weights.items():
        w    = weight * np.random.uniform(0.88, 1.12)
        ph   = np.random.uniform(0, 2*np.pi)
        hph  = np.random.uniform(0, 2*np.pi)
        data[:, ch] += (
            signal_amp * w * np.sin(2*np.pi*target_hz*t + ph)
            + signal_amp * w * 0.38 * np.sin(2*np.pi*harmonic*t + hph)
            + signal_amp * w * 0.12 * np.sin(2*np.pi*(target_hz*3)*t + np.random.uniform(0,2*np.pi))
        )

    for ch in SUPPORT_CHANNELS:
        data[:, ch] += signal_amp * sup_w * np.sin(2*np.pi*target_hz*t + np.random.uniform(0,2*np.pi))

    return data, snr_db


def extract_features(epoch):
    """
    Extract 19 SSVEP features from one Unicorn epoch.
    Uses Pz, PO7, Oz, PO8 — the parieto-occipital SSVEP channels.
    Must match EXACTLY between training and live inference.
    """
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0/FS)

    def band_power(fft_mag, target_hz, bw=0.4):
        mask = (freqs >= target_hz - bw) & (freqs <= target_hz + bw)
        return float(np.mean(fft_mag[mask] ** 2)) if mask.any() else 1e-10

    features    = []
    p_2hz_all   = []
    p_10hz_all  = []

    for ch in SSVEP_CHANNELS:   # Pz, PO7, Oz, PO8
        fft = np.abs(np.fft.rfft(epoch[:, ch]))
        p2  = band_power(fft, FREQ_BASELINE)
        p4  = band_power(fft, FREQ_BASELINE * 2)
        p10 = band_power(fft, FREQ_SLAP)
        p20 = band_power(fft, FREQ_SLAP * 2)
        features.extend([p2, p4, p10, p20])
        p_2hz_all.append(p2)
        p_10hz_all.append(p10)

    mean_p2  = max(np.mean(p_2hz_all),  1e-10)
    mean_p10 = max(np.mean(p_10hz_all), 1e-10)
    features.append(mean_p10 / mean_p2)
    features.append(np.log(mean_p10 / mean_p2 + 1e-6))

    # PO7 vs PO8 bilateral coherence at 10 Hz
    fft_po7 = np.abs(np.fft.rfft(epoch[:, 5]))
    fft_po8 = np.abs(np.fft.rfft(epoch[:, 7]))
    coh     = (band_power(fft_po7, FREQ_SLAP) * band_power(fft_po8, FREQ_SLAP)) ** 0.5 / mean_p10
    features.append(coh)

    return np.array(features, dtype=np.float32)


def generate_session(session_id, n_trials=100, snr_range=(7, 20)):
    rows = []
    for i in range(n_trials):
        for label, label_int in [('baseline', 0), ('slap', 1)]:
            snr = np.random.uniform(*snr_range)
            epoch, actual_snr = simulate_unicorn_epoch(label, snr_db=snr)
            features = extract_features(epoch)
            row = {
                'session_id': session_id, 'trial': i,
                'label': label, 'label_int': label_int,
                'snr_db': round(actual_snr, 2),
                'device': 'Unicorn Hybrid Black', 'fs': FS,
            }
            for j, f in enumerate(features):
                row[f'feat_{j:02d}'] = round(float(f), 6)
            rows.append(row)
    return rows


def main():
    os.makedirs('training_data', exist_ok=True)

    SESSION_CONFIGS = [
        {'id': 'unicorn_session_001', 'snr': (12, 22), 'n': 100, 'desc': 'Strong — gel applied'},
        {'id': 'unicorn_session_002', 'snr': (8,  18), 'n': 100, 'desc': 'Good — dry electrodes'},
        {'id': 'unicorn_session_003', 'snr': (14, 24), 'n': 100, 'desc': 'Excellent — well settled'},
        {'id': 'unicorn_session_004', 'snr': (6,  16), 'n': 100, 'desc': 'Noisier — movement artifacts'},
        {'id': 'unicorn_session_005', 'snr': (10, 20), 'n': 100, 'desc': 'Typical — mixed conditions'},
    ]

    all_rows = []
    for cfg in SESSION_CONFIGS:
        print(f"Generating {cfg['id']} ({cfg['desc']})...")
        rows = generate_session(cfg['id'], n_trials=cfg['n'], snr_range=cfg['snr'])
        all_rows.extend(rows)
        path = f"training_data/{cfg['id']}.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  → {len(rows)} trials → {path}")

    combined_path = 'training_data/all_sessions_combined.csv'
    with open(combined_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    n_feat = len([k for k in all_rows[0] if k.startswith('feat_')])
    print(f"\n✓ Combined: {len(all_rows)} trials, {n_feat} features → {combined_path}")
    print(f"  Baseline: {sum(1 for r in all_rows if r['label']=='baseline')}")
    print(f"  Slap:     {sum(1 for r in all_rows if r['label']=='slap')}")

    meta = {
        'generated': datetime.now().isoformat(),
        'device': 'Unicorn Hybrid Black',
        'fs': FS, 'epoch_sec': EPOCH_SEC, 'n_samples': N_SAMPLES,
        'channels': CHANNELS,
        'ssvep_channels': [CHANNELS[i] for i in SSVEP_CHANNELS],
        'ssvep_ch_indices': SSVEP_CHANNELS,
        'freq_baseline_hz': FREQ_BASELINE, 'freq_slap_hz': FREQ_SLAP,
        'n_features': n_feat,
        'sessions': [c['id'] for c in SESSION_CONFIGS],
        'total_trials': len(all_rows),
        'lsl_notes': 'UnicornLSL.exe streams 17ch; use only ch0-7 (EEG). Stream type=EEG, name=UN-XXXX.XX.XX',
    }
    with open('training_data/metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → training_data/metadata.json")


if __name__ == '__main__':
    main()
