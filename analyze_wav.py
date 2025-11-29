#!/usr/bin/env python
"""
Analyze WAV file with FFT to detect Doppler shifts from cars driving by.
Based on techniques from stream_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, fft
from scipy.io import wavfile
from scipy.interpolate import interp1d
import os
import math

# Constants (similar to stream_visualizer.py)
SPECTROGRAM_LENGTH = 500  # Number of FFT frames in spectrogram
FFT_LGTH_TRUNCATE = 150   # How many frequency bins to display

# Doppler radar constants
# For 24.125 GHz radar: ~161 Hz per m/s
HZ_PER_MPS = 161.0
MPS_TO_MPH = 2.23693629

def load_wav_file(filepath):
    """Load WAV file and return sample rate and audio data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"WAV file not found: {filepath}")
    
    sample_rate, audio_data = wavfile.read(filepath)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Convert to float32 and normalize
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    
    return sample_rate, audio_data

def compute_fft_spectrum(data, sample_rate, window_size=1024):
    """Compute FFT spectrum of audio data."""
    # Remove DC offset
    data = data - np.mean(data)
    
    # Apply windowing
    window = signal.windows.hann(window_size)
    
    # Compute FFT
    fft_result = fft.fft(data[:window_size] * window)
    magnitude = np.abs(fft_result[:window_size // 2])
    
    # Frequency axis
    frequencies = np.linspace(0, sample_rate / 2, window_size // 2)
    
    return frequencies, magnitude

def compute_spectrogram(data, sample_rate, nperseg=1024, noverlap=512):
    """Compute spectrogram using short-time FFT."""
    frequencies, times, Sxx = signal.spectrogram(
        data,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann',
        scaling='density'
    )
    return frequencies, times, Sxx

def estimate_noise_spectrum(data, sample_rate, noise_duration=0.1):
    """
    Estimate noise spectrum from a quiet portion of the signal.
    
    Args:
        data: Input audio signal
        sample_rate: Sample rate in Hz
        noise_duration: Duration in seconds to use for noise estimation (from start)
    
    Returns:
        Noise power spectrum
    """
    noise_samples = int(noise_duration * sample_rate)
    noise_samples = min(noise_samples, len(data) // 4)  # Use at most 25% of signal
    
    # Use the first portion (typically quiet) or find the quietest portion
    noise_segment = data[:noise_samples]
    
    # Compute power spectrum of noise
    nperseg = min(2048, len(noise_segment))
    frequencies, psd = signal.welch(noise_segment, sample_rate, nperseg=nperseg, 
                                    window='hann', scaling='density')
    
    return frequencies, psd

def spectral_subtraction(data, sample_rate, alpha=2.0, beta=0.01, noise_duration=0.1):
    """
    Reduce white noise using spectral subtraction.
    
    Args:
        data: Input audio signal
        sample_rate: Sample rate in Hz
        alpha: Over-subtraction factor (default 2.0, higher = more aggressive)
        beta: Spectral floor factor (default 0.01, prevents over-subtraction artifacts)
        noise_duration: Duration in seconds to use for noise estimation
    
    Returns:
        Denoised audio signal
    """
    # Estimate noise spectrum
    noise_freqs, noise_psd = estimate_noise_spectrum(data, sample_rate, noise_duration)
    
    # Process signal in overlapping windows
    nperseg = 2048
    noverlap = nperseg // 2
    window = signal.windows.hann(nperseg)
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(data, sample_rate, nperseg=nperseg, 
                                          noverlap=noverlap, window=window,
                                          return_onesided=True, boundary='zeros')
    
    # Interpolate noise spectrum to match STFT frequencies
    noise_interp = interp1d(noise_freqs, noise_psd, kind='linear', 
                            bounds_error=False, fill_value=noise_psd[-1])
    noise_psd_stft = noise_interp(frequencies)
    
    # Compute power spectrum
    power_spectrum = np.abs(Zxx) ** 2
    
    # Spectral subtraction
    # Subtract alpha * noise_psd, but keep at least beta * original
    denoised_power = power_spectrum - alpha * noise_psd_stft[:, np.newaxis]
    denoised_power = np.maximum(denoised_power, beta * power_spectrum)
    
    # Reconstruct magnitude with phase from original
    denoised_magnitude = np.sqrt(denoised_power)
    denoised_Zxx = denoised_magnitude * np.exp(1j * np.angle(Zxx))
    
    # Inverse STFT to reconstruct signal
    _, denoised_data = signal.istft(denoised_Zxx, sample_rate, nperseg=nperseg,
                                    noverlap=noverlap, window=window, boundary=True)
    
    # Trim to original length
    denoised_data = denoised_data[:len(data)]
    
    return denoised_data

def wiener_filter(data, sample_rate, noise_duration=0.1):
    """
    Reduce white noise using Wiener filtering (optimal in MSE sense).
    
    Args:
        data: Input audio signal
        sample_rate: Sample rate in Hz
        noise_duration: Duration in seconds to use for noise estimation
    
    Returns:
        Denoised audio signal
    """
    # Estimate noise spectrum
    noise_freqs, noise_psd = estimate_noise_spectrum(data, sample_rate, noise_duration)
    
    # Process signal in overlapping windows
    nperseg = 2048
    noverlap = nperseg // 2
    window = signal.windows.hann(nperseg)
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(data, sample_rate, nperseg=nperseg,
                                          noverlap=noverlap, window=window,
                                          return_onesided=True, boundary='zeros')
    
    # Interpolate noise spectrum to match STFT frequencies
    noise_interp = interp1d(noise_freqs, noise_psd, kind='linear',
                            bounds_error=False, fill_value=noise_psd[-1])
    noise_psd_stft = noise_interp(frequencies)
    
    # Estimate signal power spectrum (use full signal)
    signal_psd = np.mean(np.abs(Zxx) ** 2, axis=1)
    
    # Wiener filter: H(f) = S(f) / (S(f) + N(f))
    # where S is signal power and N is noise power
    epsilon = 1e-10  # Prevent division by zero
    wiener_gain = signal_psd / (signal_psd + noise_psd_stft + epsilon)
    
    # Apply Wiener filter
    wiener_gain = np.clip(wiener_gain, 0, 1)  # Ensure gain is between 0 and 1
    denoised_Zxx = Zxx * wiener_gain[:, np.newaxis]
    
    # Inverse STFT to reconstruct signal
    _, denoised_data = signal.istft(denoised_Zxx, sample_rate, nperseg=nperseg,
                                    noverlap=noverlap, window=window, boundary=True)
    
    # Trim to original length
    denoised_data = denoised_data[:len(data)]
    
    return denoised_data

def apply_bandpass_filter(data, sample_rate, lowcut=100, highcut=2500, order=4):
    """
    Apply a band-pass filter to remove noise outside the Doppler frequency range.
    
    Args:
        data: Input audio signal
        sample_rate: Sample rate in Hz
        lowcut: Low frequency cutoff in Hz (default 100 Hz to remove DC and low-frequency noise)
        highcut: High frequency cutoff in Hz (default 2500 Hz)
        order: Filter order (higher = sharper cutoff, default 4)
    
    Returns:
        Filtered audio signal
    """
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth band-pass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

def remove_time_segment(data, sample_rate, start_time, end_time, fade=True, fade_duration=0.1):
    """
    Remove or zero out a time segment from the audio signal.
    
    Args:
        data: Input audio signal
        sample_rate: Sample rate in Hz
        start_time: Start time in seconds to remove
        end_time: End time in seconds to remove
        fade: Whether to apply fade in/out at boundaries (default True)
        fade_duration: Duration of fade in/out in seconds (default 0.1)
    
    Returns:
        Audio signal with specified segment removed/zeroed
    """
    data = data.copy()  # Don't modify original
    duration = len(data) / sample_rate
    
    # Clamp times to valid range
    start_time = max(0, min(start_time, duration))
    end_time = max(start_time, min(end_time, duration))
    
    # Convert times to sample indices
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)
    
    if fade:
        # Apply fade out before removal
        fade_samples = int(fade_duration * sample_rate)
        fade_start = max(0, start_idx - fade_samples)
        fade_end = start_idx
        
        if fade_end > fade_start:
            fade_curve = np.linspace(1.0, 0.0, fade_end - fade_start)
            data[fade_start:fade_end] *= fade_curve
        
        # Zero out the segment
        data[start_idx:end_idx] = 0.0
        
        # Apply fade in after removal
        fade_start = end_idx
        fade_end = min(len(data), end_idx + fade_samples)
        
        if fade_end > fade_start:
            fade_curve = np.linspace(0.0, 1.0, fade_end - fade_start)
            data[fade_start:fade_end] *= fade_curve
    else:
        # Simply zero out the segment
        data[start_idx:end_idx] = 0.0
    
    return data

def save_wav_file(data, sample_rate, output_filepath):
    """
    Save audio data to a WAV file.
    
    Args:
        data: Audio signal (float32, normalized to [-1, 1])
        sample_rate: Sample rate in Hz
        output_filepath: Path to output WAV file
    """
    # Clip to valid range
    data = np.clip(data, -1.0, 1.0)
    
    # Convert to int16
    data_int16 = (data * 32767.0).astype(np.int16)
    
    # Save to WAV file
    wavfile.write(output_filepath, sample_rate, data_int16)
    print(f"Saved processed audio to: {output_filepath}")

def hz_to_mph(fd_hz, angle_degrees=0.0):
    """
    Convert Doppler frequency shift to speed in mph.
    
    Args:
        fd_hz: Doppler frequency shift in Hz
        angle_degrees: Angle between target motion and line-of-sight (0° = directly toward/away)
                       Default 0° assumes direct line-of-sight motion.
    
    Returns:
        Speed in mph
    
    Physics:
        Doppler shift measures radial velocity: v_r = fd / (2 * f0 / c)
        If target moves at angle θ from line-of-sight, radial velocity is: v_r = v * cos(θ)
        Therefore actual speed: v = v_r / cos(θ)
        For 45° angle: v = v_r / cos(45°) = v_r / 0.707 ≈ v_r * 1.414
    """
    # Calculate radial velocity from Doppler shift (m/s)
    radial_velocity_mps = fd_hz / HZ_PER_MPS
    
    # If angle is 0, no correction needed (direct line-of-sight)
    if angle_degrees == 0.0:
        actual_velocity_mps = radial_velocity_mps
    else:
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        # Actual speed = radial_velocity / cos(angle)
        # Handle edge case where angle approaches 90 degrees
        cos_angle = math.cos(angle_rad)
        if abs(cos_angle) < 1e-10:  # Avoid division by zero
            cos_angle = 1e-10 if cos_angle >= 0 else -1e-10
        actual_velocity_mps = radial_velocity_mps / cos_angle
    
    return actual_velocity_mps * MPS_TO_MPH

def analyze_wav(filepath, noise_reduction='wiener', noise_reduction_enabled=True,
                remove_time_segment_start=None, remove_time_segment_end=None, 
                save_output=True, angle_degrees=0.0):
    """
    Main analysis function.
    
    Args:
        filepath: Path to WAV file
        noise_reduction: Method to use ('wiener', 'spectral_subtraction', or None)
        noise_reduction_enabled: Whether to apply noise reduction (default True)
        remove_time_segment_start: Start time in seconds to remove (None to disable)
        remove_time_segment_end: End time in seconds to remove
        save_output: Whether to save processed audio to WAV file (default True)
        angle_degrees: Angle between target motion and line-of-sight in degrees (0° = direct, default 0.0)
                       Use 45° if targets are moving at 45 degrees to the sensor.
    """
    print(f"Loading WAV file: {filepath}")
    sample_rate, audio_data = load_wav_file(filepath)
    
    duration = len(audio_data) / sample_rate
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Number of samples: {len(audio_data)}")
    
    # Remove time segment if specified (before noise reduction)
    if remove_time_segment_start is not None and remove_time_segment_end is not None:
        print(f"Removing time segment {remove_time_segment_start:.2f}s - {remove_time_segment_end:.2f}s...")
        audio_data = remove_time_segment(audio_data, sample_rate, 
                                        remove_time_segment_start, remove_time_segment_end)
    
    # Apply noise reduction if enabled
    if noise_reduction_enabled and noise_reduction:
        if noise_reduction == 'wiener':
            print("Applying Wiener filter for white noise reduction...")
            audio_data = wiener_filter(audio_data, sample_rate)
        elif noise_reduction == 'spectral_subtraction':
            print("Applying spectral subtraction for white noise reduction...")
            audio_data = spectral_subtraction(audio_data, sample_rate)
        else:
            print(f"Unknown noise reduction method: {noise_reduction}, skipping...")
    
    # Apply band-pass filter to reduce noise
    print("Applying band-pass filter (100-2500 Hz)...")
    audio_data_filtered = apply_bandpass_filter(audio_data, sample_rate, lowcut=100, highcut=2500)
    
    # Create figure with subplots - spectrogram gets 50% of space
    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 2, 0], hspace=2.0)
    
    # 1. Time domain plot (filtered signal) - top 25%
    ax1 = plt.subplot(gs[0])
    time_axis = np.linspace(0, duration, len(audio_data_filtered))
    ax1.plot(time_axis, audio_data_filtered)
    
    # Highlight removed time segment if specified
    if remove_time_segment_start is not None and remove_time_segment_end is not None:
        y_min, y_max = ax1.get_ylim()
        ax1.axvspan(remove_time_segment_start, remove_time_segment_end, 
                   alpha=0.3, color='red', label='Removed segment')
        ax1.legend()
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    title = 'Filtered Audio Signal (Time Domain'
    if remove_time_segment_start is not None and remove_time_segment_end is not None:
        title += f', removed {remove_time_segment_start:.1f}s-{remove_time_segment_end:.1f}s'
    if noise_reduction_enabled and noise_reduction:
        title += f', {noise_reduction} noise reduction'
    if angle_degrees != 0.0:
        title += f', {angle_degrees:.0f}° angle correction'
    title += ', 100-2500 Hz band-pass)'
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # 2. Frequency spectrum (overall) - second 25%
    ax2 = plt.subplot(gs[1])
    window_size = min(4096, len(audio_data_filtered) // 4)
    freqs, magnitude = compute_fft_spectrum(audio_data_filtered, sample_rate, window_size)
    
    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Filter out DC and very low frequencies
    valid_idx = freqs > 100  # Ignore frequencies below 100 Hz
    freqs_filtered = freqs[valid_idx]
    magnitude_filtered = magnitude_db[valid_idx]
    
    ax2.plot(freqs_filtered, magnitude_filtered, 'b-', linewidth=0.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Frequency Spectrum (Full Signal)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, min(2500, sample_rate / 2)])
    
    # Add speed scale on top axis (for Doppler interpretation)
    ax2_top = ax2.twiny()
    speed_axis = np.array([hz_to_mph(f, angle_degrees) for f in freqs_filtered])
    ax2_top.set_xlim(ax2.get_xlim())
    speed_label = 'Speed (mph)'
    if angle_degrees != 0.0:
        speed_label += f' (angle-corrected, {angle_degrees:.0f}°)'
    ax2_top.set_xlabel(speed_label)
    
    # 3. Spectrogram - bottom 50%
    ax3 = plt.subplot(gs[2])
    nperseg = min(2048, len(audio_data_filtered) // 20)
    noverlap = nperseg // 2
    frequencies, times, Sxx = compute_spectrogram(audio_data_filtered, sample_rate, nperseg, noverlap)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Limit frequency range for better visualization
    max_freq_show = min(2500, sample_rate / 2)
    freq_mask = frequencies <= max_freq_show
    frequencies_show = frequencies[freq_mask]
    Sxx_show = Sxx_db[freq_mask, :]
    
    im = ax3.pcolormesh(times, frequencies_show, Sxx_show, shading='gouraud', cmap='hot')
    
    # Highlight removed time segment if specified
    if remove_time_segment_start is not None and remove_time_segment_end is not None:
        freq_min, freq_max = ax3.get_ylim()
        ax3.axvspan(remove_time_segment_start, remove_time_segment_end, 
                   alpha=0.3, color='red', label='Removed segment')
        ax3.legend()
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Spectrogram (Time-Frequency Analysis)')
    plt.colorbar(im, ax=ax3, label='Power (dB)')
    
    # Add speed scale on right axis
    ax3_right = ax3.twinx()
    speed_max = hz_to_mph(frequencies_show[-1], angle_degrees)
    ax3_right.set_ylim(ax3.get_ylim())
    speed_label = 'Speed (mph)'
    if angle_degrees != 0.0:
        speed_label += f' (angle-corrected, {angle_degrees:.0f}°)'
    ax3_right.set_ylabel(speed_label)
    # Set speed ticks on the right axis
    freq_ticks = ax3.get_yticks()
    speed_ticks = np.array([hz_to_mph(f, angle_degrees) for f in freq_ticks])
    ax3_right.set_yticks(freq_ticks)
    ax3_right.set_yticklabels([f'{s:.1f}' for s in speed_ticks])
    
    plt.tight_layout(pad=3.0)
    
    # Save the figure
    output_file = filepath.replace('.wav', '_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plot to: {output_file}")
    
    # Print statistics
    print("\n=== Analysis Statistics ===")
    if angle_degrees != 0.0:
        print(f"Angle correction: {angle_degrees:.1f}° (correction factor: {1.0/math.cos(math.radians(angle_degrees)):.3f}x)")
    peak_freq = freqs_filtered[np.argmax(magnitude_filtered)]
    print(f"Peak frequency in overall spectrum: {peak_freq:.2f} Hz")
    print(f"Equivalent speed: {hz_to_mph(peak_freq, angle_degrees):.2f} mph")
    
    # Find peaks in frequency domain
    peaks, properties = signal.find_peaks(magnitude_filtered, height=np.max(magnitude_filtered) - 20)
    if len(peaks) > 0:
        print(f"\nFound {len(peaks)} significant frequency peaks:")
        for i, peak_idx in enumerate(peaks[:10]):  # Show top 10
            freq = freqs_filtered[peak_idx]
            mag = magnitude_filtered[peak_idx]
            speed = hz_to_mph(freq, angle_degrees)
            print(f"  Peak {i+1}: {freq:.2f} Hz ({speed:.2f} mph) - magnitude: {mag:.2f} dB")
    
    # Analyze spectrogram for time-varying patterns
    print("\n=== Spectrogram Analysis ===")
    # Find peaks in each time slice
    max_speeds = []
    for t_idx in range(Sxx_show.shape[1]):
        time_slice = Sxx_show[:, t_idx]
        if np.max(time_slice) > np.min(time_slice) + 10:  # Significant signal
            peak_idx = np.argmax(time_slice)
            freq_peak = frequencies_show[peak_idx]
            speed = hz_to_mph(freq_peak, angle_degrees)
            max_speeds.append((times[t_idx], speed, freq_peak))
    
    if max_speeds:
        speeds_only = [s[1] for s in max_speeds]
        print(f"Detected speed range: {min(speeds_only):.1f} - {max(speeds_only):.1f} mph")
        print(f"Average peak speed: {np.mean(speeds_only):.1f} mph")
        print(f"Max detected speed: {max(speeds_only):.1f} mph")
    
    # Save processed audio to WAV file
    if save_output:
        output_wav = filepath.replace('.wav', '_processed.wav')
        save_wav_file(audio_data_filtered, sample_rate, output_wav)
    
    plt.show()
    
    return {
        'sample_rate': sample_rate,
        'duration': duration,
        'frequencies': freqs_filtered,
        'magnitude': magnitude_filtered,
        'spectrogram': (frequencies_show, times, Sxx_show),
        'peak_speeds': max_speeds,
        'processed_audio': audio_data_filtered
    }

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze WAV file with FFT to detect Doppler shifts from cars driving by.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Noise reduction options:
  wiener              - Wiener filter (optimal, recommended)
  spectral_subtraction - Spectral subtraction (more aggressive)
  none                - Disable noise reduction
        """
    )
    parser.add_argument('wav_file', nargs='?', default='cars-orig.wav',
                       help='Path to WAV file (default: cars-orig.wav)')
    parser.add_argument('--noise-reduction', '-n',
                       choices=['wiener', 'spectral_subtraction', 'none'],
                       default='wiener',
                       help='Noise reduction method (default: wiener)')
    parser.add_argument('--remove-start', type=float, default=None,
                       help='Start time in seconds to remove from signal (e.g., 16.0)')
    parser.add_argument('--remove-end', type=float, default=None,
                       help='End time in seconds to remove from signal (e.g., 22.0)')
    parser.add_argument('--angle', type=float, default=45.0,
                       help='Angle between target motion and line-of-sight in degrees (default: 45.0)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save processed audio to WAV file')
    
    args = parser.parse_args()
    wav_file = args.wav_file
    
    if not os.path.exists(wav_file):
        print(f"Error: File not found: {wav_file}")
        print(f"Looking in: {os.path.abspath(os.path.dirname(wav_file))}")
        sys.exit(1)
    
    noise_reduction = None if args.noise_reduction == 'none' else args.noise_reduction
    noise_reduction_enabled = args.noise_reduction != 'none'
    
    # Validate time removal parameters
    remove_start = args.remove_start
    remove_end = args.remove_end
    if (remove_start is not None and remove_end is None) or \
       (remove_start is None and remove_end is not None):
        print("Error: Both --remove-start and --remove-end must be specified together")
        sys.exit(1)
    
    if remove_start is not None and remove_end is not None:
        if remove_start >= remove_end:
            print("Error: --remove-start must be less than --remove-end")
            sys.exit(1)
    
    # Validate angle parameter
    angle_degrees = args.angle
    if angle_degrees < 0 or angle_degrees >= 90:
        print("Warning: Angle should be between 0 and 90 degrees. Using absolute value.")
        angle_degrees = abs(angle_degrees) % 90
    
    try:
        results = analyze_wav(wav_file, 
                             noise_reduction=noise_reduction, 
                             noise_reduction_enabled=noise_reduction_enabled,
                             remove_time_segment_start=remove_start,
                             remove_time_segment_end=remove_end,
                             save_output=not args.no_save,
                             angle_degrees=angle_degrees)
        print("\nAnalysis complete!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

