"""
NeuroJack — Unicorn Hybrid Black Real-Time EEG WebSocket Server
===============================================================
Bridges the Unicorn Hybrid Black (via UnicornLSL) to the browser game.

HOW TO START STREAMING FROM YOUR UNICORN:
  1. Connect the Unicorn Bluetooth dongle to your PC
  2. Power on the Unicorn headset (LED blinks → solid when connected)
  3. Open Unicorn Suite → launch "UnicornLSL" application
  4. In UnicornLSL: click "Start" to begin LSL streaming
  5. Run this server: python eeg_server.py
  6. Open NeuroJack_EEG.html in Chrome/Firefox

LSL Stream details from UnicornLSL:
  - Stream name: "UN-XXXX.XX.XX" (your device serial number)
  - Stream type: "EEG"
  - Channel count: 17
    Ch 0-7:   EEG (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8) in microvolts
    Ch 8-10:  Accelerometer X, Y, Z
    Ch 11-13: Gyroscope X, Y, Z
    Ch 14:    Battery Level
    Ch 15:    Counter
    Ch 16:    Validation Indicator
  - Sample rate: 250 Hz
  - We use ONLY channels 0-7 (EEG), specifically 4-7 (Pz, PO7, Oz, PO8) for SSVEP
"""

import asyncio
import json
import pickle
import numpy as np
import websockets
import threading
import time
import queue
import traceback
import os
import sys
from collections import deque

# Import EXACT same feature extractor used during training
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_training_data import (
    extract_features,
    FS,                # 250
    N_SAMPLES,         # 250
    CHANNELS,          # ['Fz','C3','Cz','C4','Pz','PO7','Oz','PO8']
    SSVEP_CHANNELS,    # [4, 5, 6, 7]
    FREQ_BASELINE,     # 2.0
    FREQ_SLAP,         # 10.0
)

# ── Config ────────────────────────────────────────────────────────────────────
WS_HOST         = 'localhost'
WS_PORT         = 8765
EPOCH_SEC       = 1.0         # classification window (250 samples @ 250 Hz)
STEP_SEC        = 0.25        # slide step (4 predictions/sec)
EPOCH_N         = int(FS * EPOCH_SEC)    # 250
STEP_N          = int(FS * STEP_SEC)     # 62

CONFIDENCE_THR  = 0.65        # minimum p_slap to trigger game slap
SMOOTH_N        = 4           # smooth over this many consecutive predictions

# Unicorn LSL stream identifiers
UNICORN_STREAM_TYPE = 'EEG'
UNICORN_N_CHANNELS  = 17      # full LSL stream width from UnicornLSL
UNICORN_EEG_SLICE   = slice(0, 8)   # first 8 channels are EEG


# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    path = 'model/ssvep_classifier.pkl'
    if not os.path.exists(path):
        print("❌ Model not found. Run: python train_classifier.py")
        sys.exit(1)
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"✓ SSVEP classifier loaded")
    print(f"  Trained on {bundle['trained_on']} trials")
    print(f"  Training accuracy: {bundle['accuracy']:.1%}")
    return bundle


# ── Unicorn LSL Reader ────────────────────────────────────────────────────────
class UnicornLSLReader:
    """
    Reads from the Unicorn LSL stream (via UnicornLSL.exe).
    Falls back to simulation if no stream found.
    """

    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.running    = False

    def find_unicorn_stream(self):
        try:
            from pylsl import resolve_byprop
        except ImportError:
            print("⚠  pylsl not installed: pip install pylsl")
            return None

        print("🔍 Searching for Unicorn LSL stream (type='EEG')...")
        print("   Make sure UnicornLSL.exe is running and streaming!")
        streams = resolve_byprop('type', UNICORN_STREAM_TYPE, timeout=8.0)

        if not streams:
            print("⚠  No Unicorn stream found — switching to simulation mode")
            print("   To use real EEG: open Unicorn Suite → UnicornLSL → Start")
            return None

        # Prefer stream whose name starts with 'UN-' (Unicorn serial number format)
        unicorn_streams = [s for s in streams if s.name().startswith('UN-')]
        stream = unicorn_streams[0] if unicorn_streams else streams[0]

        print(f"✓ Unicorn stream found!")
        print(f"  Name:     {stream.name()}")
        print(f"  Channels: {stream.channel_count()} (using first 8: EEG)")
        print(f"  Rate:     {stream.nominal_srate()} Hz")
        return stream

    def run(self):
        self.running = True
        stream = self.find_unicorn_stream()

        if stream is None:
            self._run_simulation()
            return

        try:
            from pylsl import StreamInlet
            inlet = StreamInlet(stream, max_buflen=30)
            actual_fs = inlet.info().nominal_srate()
            n_ch      = inlet.info().channel_count()

            print(f"\n🧠 Unicorn live EEG streaming at {actual_fs} Hz")

            while self.running:
                # Pull up to 32 samples at once (128ms chunks)
                samples, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=32)
                if samples:
                    full_chunk = np.array(samples, dtype=np.float32)  # shape: (n, 17)
                    # Extract only EEG channels 0-7
                    eeg_chunk  = full_chunk[:, UNICORN_EEG_SLICE]     # shape: (n, 8)
                    self.data_queue.put(('eeg', eeg_chunk, float(actual_fs)))

        except Exception as e:
            print(f"LSL stream error: {e}")
            traceback.print_exc()
            print("Falling back to simulation...")
            self._run_simulation()

    def _run_simulation(self):
        """
        Simulates Unicorn EEG at 250 Hz.
        Alternates between baseline (2 Hz SSVEP) and slap (10 Hz SSVEP) states
        every 5 seconds so you can watch the classifier respond.
        """
        from generate_training_data import simulate_unicorn_epoch

        print("\n🎮 SIMULATION MODE — No Unicorn connected")
        print("   To use real EEG:")
        print("   1. Connect Unicorn dongle + power on headset")
        print("   2. Open Unicorn Suite → UnicornLSL → Start")
        print("   3. Restart this server\n")

        CHUNK  = 25             # 25 samples = 100ms chunks at 250 Hz
        SLEEP  = CHUNK / FS     # sleep between chunks

        t           = 0.0
        state       = 'baseline'
        state_start = time.time()
        STATE_DUR   = 5.0       # seconds per state

        # Pre-generate epochs and stream chunk-by-chunk
        current_epoch, _ = simulate_unicorn_epoch(state)
        sample_idx = 0

        while self.running:
            now = time.time()
            if now - state_start >= STATE_DUR:
                state       = 'slap' if state == 'baseline' else 'baseline'
                state_start = now
                sample_idx  = 0
                current_epoch, _ = simulate_unicorn_epoch(state, snr_db=16)
                print(f"[SIM] State → {state.upper()} ({FREQ_SLAP if state=='slap' else FREQ_BASELINE} Hz SSVEP)")

            # Stream the current epoch in chunks (loop if needed)
            end_idx = min(sample_idx + CHUNK, N_SAMPLES)
            chunk   = current_epoch[sample_idx:end_idx]

            # If chunk exhausted, regenerate epoch
            if len(chunk) == 0:
                current_epoch, _ = simulate_unicorn_epoch(state, snr_db=16)
                sample_idx = 0
                chunk      = current_epoch[:CHUNK]

            self.data_queue.put(('eeg', chunk.astype(np.float32), float(FS)))
            sample_idx += len(chunk)
            if sample_idx >= N_SAMPLES:
                sample_idx = 0
                current_epoch, _ = simulate_unicorn_epoch(state, snr_db=16)

            time.sleep(SLEEP)

    def stop(self):
        self.running = False


# ── Sliding-Window Classifier ─────────────────────────────────────────────────
class SSVEPClassifier:
    """Rolling buffer + epoch extraction + classification."""

    def __init__(self, clf, result_queue, loop):
        self.clf          = clf
        self.result_queue = result_queue
        self.loop         = loop
        self.buffer       = deque()
        self.buf_len      = 0
        self.conf_history = deque(maxlen=SMOOTH_N)

    def push(self, chunk, fs):
        """Add EEG chunk. Resample if headset rate differs from expected."""
        if abs(fs - FS) > 2:
            factor  = FS / fs
            n_new   = max(1, int(len(chunk) * factor))
            indices = np.linspace(0, len(chunk)-1, n_new).astype(int)
            chunk   = chunk[indices]

        self.buffer.append(chunk)
        self.buf_len += len(chunk)

        while self.buf_len >= EPOCH_N:
            epoch = np.concatenate(list(self.buffer), axis=0)[-EPOCH_N:]
            self._classify(epoch)

            # Drop one step
            to_drop = STEP_N
            while to_drop > 0 and self.buffer:
                bl = len(self.buffer[0])
                if bl <= to_drop:
                    self.buf_len -= bl
                    to_drop      -= bl
                    self.buffer.popleft()
                else:
                    self.buffer[0] = self.buffer[0][to_drop:]
                    self.buf_len  -= to_drop
                    to_drop        = 0

    def _classify(self, epoch):
        try:
            feats  = extract_features(epoch).reshape(1, -1)
            proba  = self.clf.predict_proba(feats)[0]  # [p_baseline, p_slap]

            self.conf_history.append(proba[1])
            smooth_p_slap = float(np.mean(self.conf_history))

            result = {
                'prediction': 'slap' if smooth_p_slap >= CONFIDENCE_THR else 'baseline',
                'p_slap':     round(smooth_p_slap, 3),
                'p_baseline': round(1 - smooth_p_slap, 3),
                'raw_p_slap': round(float(proba[1]), 3),
                'confidence': round(float(max(proba)), 3),
                'trigger':    smooth_p_slap >= CONFIDENCE_THR,
            }
            asyncio.run_coroutine_threadsafe(
                self.result_queue.put(result), self.loop
            )
        except Exception:
            pass


# ── WebSocket Server ──────────────────────────────────────────────────────────
connected_clients = set()


async def ws_handler(websocket):
    connected_clients.add(websocket)
    client_id = id(websocket)
    print(f"🌐 Browser connected  [total: {len(connected_clients)}]")

    try:
        await websocket.send(json.dumps({
            'type':    'connected',
            'message': 'Unicorn Hybrid Black EEG server ready',
            'device':  'Unicorn Hybrid Black',
            'fs':       FS,
            'channels': CHANNELS,
            'ssvep_channels': [CHANNELS[i] for i in SSVEP_CHANNELS],
            'confidence_threshold': CONFIDENCE_THR,
        }))
        async for msg in websocket:
            try:
                data = json.loads(msg)
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"🌐 Browser disconnected [total: {len(connected_clients)}]")


async def broadcast_loop(result_queue):
    while True:
        result = await result_queue.get()
        if connected_clients:
            msg = json.dumps({'type': 'eeg_prediction', **result})
            await asyncio.gather(
                *[c.send(msg) for c in connected_clients],
                return_exceptions=True
            )


async def main():
    bundle = load_model()

    data_queue   = queue.Queue()
    result_queue = asyncio.Queue()
    loop         = asyncio.get_running_loop()

    classifier = SSVEPClassifier(bundle['classifier'], result_queue, loop)
    reader     = UnicornLSLReader(data_queue)

    # LSL reader thread
    lsl_thread = threading.Thread(target=reader.run, daemon=True)
    lsl_thread.start()

    # Classification thread
    def classify_loop():
        while True:
            try:
                kind, chunk, fs = data_queue.get(timeout=1.0)
                if kind == 'eeg':
                    classifier.push(chunk, fs)
            except queue.Empty:
                pass
            except Exception:
                traceback.print_exc()

    clf_thread = threading.Thread(target=classify_loop, daemon=True)
    clf_thread.start()

    print(f"\n{'='*60}")
    print(f"  NeuroJack EEG Server  —  ws://{WS_HOST}:{WS_PORT}")
    print(f"  Device:  Unicorn Hybrid Black  ({FS} Hz)")
    print(f"  SSVEP:   {[CHANNELS[i] for i in SSVEP_CHANNELS]}")
    print(f"  Trigger: p_slap ≥ {CONFIDENCE_THR}  (smoothed over {SMOOTH_N} epochs)")
    print(f"{'='*60}")
    print(f"\n  Open NeuroJack_EEG.html in your browser")
    print(f"  Press Ctrl+C to stop\n")

    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await broadcast_loop(result_queue)


if __name__ == '__main__':
    asyncio.run(main())
