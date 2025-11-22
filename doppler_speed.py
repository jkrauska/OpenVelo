import numpy as np
import sounddevice as sd
import time
import math
from collections import deque

# ---------- User-tweakable settings ----------
SAMPLE_RATE = 48000          # Hz (USB audio standard)
BLOCK_SIZE  = 2048           # samples per FFT frame (~42.7 ms at 48 kHz)
FMIN = 300                   # ignore low rumble / DC / AGC breathing
FMAX = 8000                  # baseball Doppler is ~5-7 kHz for 70-90 mph
SPEED_SMOOTH_FRAMES = 5      # moving average smoothing

# 24.125 GHz radar => ~161.0 Hz per m/s (theory)
HZ_PER_MPS = 161.0
MPS_TO_MPH = 2.23693629

# optional calibration factor you can tweak after testing
CAL_FACTOR = 1.0

# Optional: set device index if needed (see list_devices() below)
DEVICE_INDEX = None
# -------------------------------------------

def list_devices():
    print(sd.query_devices())

def hz_to_mph(fd_hz):
    mps = (fd_hz / HZ_PER_MPS) * CAL_FACTOR
    return mps * MPS_TO_MPH

# Hann window reduces spectral leakage
window = np.hanning(BLOCK_SIZE)

# smoothing buffer
mph_hist = deque(maxlen=SPEED_SMOOTH_FRAMES)

def callback(indata, frames, time_info, status):
    if status:
        print(status)

    # indata is shape (frames, channels)
    x = indata[:, 0].astype(np.float32)

    # remove DC offset
    x = x - np.mean(x)

    # apply window
    xw = x * window

    # FFT
    fft = np.fft.rfft(xw)
    mag = np.abs(fft)

    freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0/SAMPLE_RATE)

    # band-pass select
    band = (freqs >= FMIN) & (freqs <= FMAX)
    band_mag = mag[band]
    band_freqs = freqs[band]

    if band_mag.size == 0:
        return

    # peak frequency
    peak_i = np.argmax(band_mag)
    peak_hz = band_freqs[peak_i]

    mph = hz_to_mph(peak_hz)
    mph_hist.append(mph)
    mph_smooth = sum(mph_hist) / len(mph_hist)

    # print one line updated in place
    print(f"\rPeak {peak_hz:7.1f} Hz  ->  {mph_smooth:5.1f} mph    ", end="")

def main():
    # Uncomment to list devices if needed:
    # list_devices(); return

    print("Starting Doppler decoder. Ctrl+C to stop.")
    print(f"Sample rate: {SAMPLE_RATE} Hz, block: {BLOCK_SIZE} samples")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=DEVICE_INDEX,
        channels=1,
        dtype="float32",
        callback=callback
    ):
        while True:
            time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
