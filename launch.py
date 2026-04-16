#!/usr/bin/env python3
"""
NeuroJack — One-Click Launcher
===============================
Generates data, trains model, and starts the EEG server.
Run this once to set everything up, then just run eeg_server.py on future sessions.
"""

import subprocess
import sys
import os
import time

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, cmd], cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n❌ Failed: {cmd}")
        sys.exit(1)
    print(f"\n✓ Done: {label}")

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    print("""
╔══════════════════════════════════════════════════╗
║         NeuroJack EEG System Launcher            ║
║         SSVEP Brain-Computer Interface           ║
╚══════════════════════════════════════════════════╝
    """)

    # Step 1: Generate training data
    if not os.path.exists('training_data/all_sessions_combined.csv'):
        run('generate_training_data.py', 'Step 1/2 — Generating synthetic EEG training data...')
    else:
        print("\n✓ Training data already exists — skipping generation")

    # Step 2: Train classifier
    if not os.path.exists('model/ssvep_classifier.pkl'):
        run('train_classifier.py', 'Step 2/2 — Training SSVEP classifier...')
    else:
        print("✓ Model already trained — skipping training")

    # Step 3: Start EEG server
    print(f"""
{'='*60}
  Step 3/3 — Starting EEG WebSocket server...
{'='*60}

Next steps:
  1. Open  NeuroJack_EEG.html  in your browser
  2. Connect your EEG headset:
       • Muse:    Start BlueMuse or mind-monitor (enable LSL)
       • OpenBCI: Open OpenBCI GUI → Networking → LSL → Start
       • Other:   Start any LSL-compatible EEG app
  3. Click START CALIBRATION in the game
  4. Your brain controls the slap!

  (No headset? The server runs in SIMULATION mode automatically)

{'='*60}
    """)

    os.execv(sys.executable, [sys.executable, 'eeg_server.py'])


if __name__ == '__main__':
    main()
