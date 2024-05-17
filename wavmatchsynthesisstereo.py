import numpy as np
import librosa
import scipy.spatial
import soundfile as sf
import argparse

def load_audio(filename):
    # Load an audio file as a floating point time series, maintaining stereo if present.
    audio, sr = librosa.load(filename, sr=None, mono=False)
    return audio, sr

def resample_audio(audio, original_sr, target_sr):
    # Resample audio to the target sample rate, handling stereo if necessary.
    if audio.ndim > 1:  # Check if audio is stereo
        audio = np.vstack([librosa.resample(audio[i], orig_sr=original_sr, target_sr=target_sr) for i in range(audio.shape[0])])
    else:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return audio, target_sr

def windowed_features(audio, window_size, hop_length, sr):
    # Compute MFCCs for each window in the audio file, handling stereo if necessary.
    if audio.ndim > 1:  # Stereo
        return [librosa.feature.mfcc(y=audio[i], sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=window_size).T for i in range(audio.shape[0])]
    else:  # Mono
        return [librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=window_size).T]

def find_best_matches(features1, features2):
    # Calculate the Euclidean distance and find the best match for each window for each channel
    return [np.argmin(scipy.spatial.distance.cdist(f1, f2, 'euclidean'), axis=1) for f1, f2 in zip(features1, features2)]

def reconstruct_audio(best_matches, audio2, window_size, hop_length):
    # Reconstruct the audio from the best matching windows for each channel
    window_function = np.hanning(window_size)  # Hann window for smooth transitions
    reconstructed_channels = []
    
    for channel_idx, channel_matches in enumerate(best_matches):
        reconstructed = np.zeros((len(channel_matches) * hop_length + window_size,))
        for i, match_idx in enumerate(channel_matches):
            start = match_idx * hop_length
            end = start + window_size
            # Ensure we handle the audio channel correctly
            actual_window = audio2[channel_idx, start:end] if audio2.ndim > 1 else audio2[start:end]
            # Apply window function to each window
            if len(actual_window) < window_size:
                actual_window = np.pad(actual_window, (0, window_size - len(actual_window)), mode='constant')
            actual_window = actual_window * window_function
            reconstructed[i * hop_length : i * hop_length + window_size] += actual_window
        reconstructed_channels.append(reconstructed)

    return np.vstack(reconstructed_channels)

def main(file1, file2, window_size, output_file):
    window_size = int(window_size)
    hop_length = window_size // 2  # Overlap of 50%

    # Load audio files
    audio1, sr1 = load_audio(file1)
    audio2, sr2 = load_audio(file2)

    # Check and resample if sample rates are different
    if sr1 != sr2:
        print(f"Resampling from {sr2} to {sr1}")
        audio1, sr1 = resample_audio(audio1, sr1, sr1)
        audio2, sr2 = resample_audio(audio2, sr2, sr1)

    # Compute features for both audio files
    features1 = windowed_features(audio1, window_size, hop_length, sr1)
    features2 = windowed_features(audio2, window_size, hop_length, sr2)

    # Find best matching windows for each channel
    best_matches = find_best_matches(features1, features2)

    # Reconstruct the audio for each channel
    reconstructed_audio = reconstruct_audio(best_matches, audio2, window_size, hop_length)

    # Save the reconstructed stereo audio
    sf.write(output_file, reconstructed_audio.T, sr1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match and reconstruct stereo audio from two WAV files.')
    parser.add_argument('file1', type=str, help='Path to the first input WAV file')
    parser.add_argument('file2', type=str, help='Path to the second input WAV file')
    parser.add_argument('window_size', type=int, help='Window size in samples')
    parser.add_argument('output_file', type=str, help='Path for the output reconstructed WAV file')
    
    args = parser.parse_args()
    
    main(args.file1, args.file2, args.window_size, args.output_file)
