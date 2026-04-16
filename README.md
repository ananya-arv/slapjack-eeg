# NeuroJack v2.0 — Unicorn Hybrid Black Edition

Real-time SSVEP brain-computer interface slapjack game powered by the g.tec Unicorn Hybrid Black.

---

## Unicorn Hybrid Black Specs (what the code is tuned for)

| Parameter | Value |
|---|---|
| Channels | Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 |
| Sample Rate | **250 Hz** |
| Resolution | 24-bit |
| Bandwidth | 0.1–60 Hz |
| SSVEP channels used | **Pz, PO7, Oz, PO8** (parieto-occipital) |
| Reference / Ground | Left & Right mastoids |

---

## Startup — Every Session

### Step 1: Start the Unicorn streaming
1. Plug in the **Bluetooth dongle**
2. Power on the **Unicorn headset** (LED blinks → goes solid)
3. Open **Unicorn Suite** → launch **UnicornLSL**
4. Click **Start** → verify the stream shows `UN-XXXX.XX.XX`

### Step 2: Run the server
```bash
python3 launch.py
```
(Or step by step: `generate_training_data.py` → `train_classifier.py` → `eeg_server.py`)

### Step 3: Open the game
Open `NeuroJack_EEG.html` in Chrome or Firefox.  
The header should show **EEG: LIVE** in green.

### Step 4: Put on the cap
- Apply gel to all 8 electrode positions (especially Pz, PO7, Oz, PO8)
- Impedance should be < 10 kΩ (check Unicorn Suite signal quality view — all green)
- Sit still, minimize jaw/eye movement during play

---

## How the SSVEP paradigm works

```
BASELINE:  You watch the flickering cards  (2 Hz) → Pz/PO7/Oz/PO8 show 2 Hz power spike
SLAP INTENT: You look at the SLAP box    (10 Hz) → same channels show 10 Hz power spike
CLASSIFIER: computes 10Hz/2Hz power ratio every 250ms → triggers game slap when ratio ≥ threshold
```

The Unicorn cap places **Oz and PO7/PO8 directly over the primary visual cortex** — ideal for SSVEP.

---

## File Structure

```
neurojack/
├── NeuroJack_EEG.html             ← Game (open in browser)
├── eeg_server.py                  ← WebSocket server (Unicorn-tuned)
├── generate_training_data.py      ← Synthetic data generator (250 Hz, 8ch)
├── train_classifier.py            ← Model trainer (LDA+SVM+RF ensemble)
├── launch.py                      ← One-click launcher
├── training_data/
│   ├── unicorn_session_00[1-5].csv  ← 1000 synthetic trials
│   ├── all_sessions_combined.csv
│   └── metadata.json
└── model/
    ├── ssvep_classifier.pkl       ← Trained model
    └── feature_importance.json
```

---

## EEG HUD indicators (in-game)

| Indicator | Meaning |
|---|---|
| **EEG: LIVE** (green) | Connected to eeg_server.py |
| **EEG: OFFLINE** (gray) | Server not running or browser can't reach it |
| **SLAP P: XX%** | Real-time probability your brain is in slap state |
| **Green bar** (top) | Visual confidence meter — fills as you shift gaze to SLAP |

Auto-slap fires when `p_slap ≥ 0.65` **and** a Jack is showing on screen.

---

## Tuning for your signal

If auto-slap isn't firing (signal too weak):
```python
# In eeg_server.py, lower this:
CONFIDENCE_THR = 0.55   # default 0.65
```

If getting false slaps (too many misfires):
```python
CONFIDENCE_THR = 0.72
SMOOTH_N = 6            # smooth over more epochs
```

---

## Collecting real training data (do this after 2-3 sessions)

1. In the game, export **Card Log CSV** (↓ CSV button) — has timestamps for every card
2. Export your Unicorn recording as CSV from Unicorn Recorder (BDF or CSV)
3. Align by timestamps: Jack events in card log = "slap" epochs in EEG; everything else = "baseline"
4. Extract 1-second epochs and replace the `training_data/` CSVs
5. Re-run `train_classifier.py` → expect 80–92% accuracy on real data

---

## Troubleshooting

| Problem | Fix |
|---|---|
| EEG shows OFFLINE | Is `eeg_server.py` running? Check port 8765 |
| UnicornLSL not found | Start UnicornLSL before running the server |
| Server shows simulation | UnicornLSL must be streaming before server starts |
| Poor impedance (red in Unicorn Suite) | Apply more gel, especially at Pz/Oz/PO7/PO8 |
| Auto-slap never fires | Lower `CONFIDENCE_THR` or check electrode contact |
| Frequent false slaps | Raise `CONFIDENCE_THR`, ensure you look at cards between rounds |

### Verify Unicorn LSL stream from Python:
```python
from pylsl import resolve_byprop
streams = resolve_byprop('type', 'EEG', timeout=5)
print([(s.name(), s.channel_count(), s.nominal_srate()) for s in streams])
# Should print: [('UN-XXXX.XX.XX', 17, 250.0)]
```
