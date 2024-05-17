# wavmatchsynthesis
This script slices two .wav files into windows and reconstructs the first file by matching its windows with those from the second using Euclidean distance calculations. Windows are modified by a Hanning function with 50% overlap. If two files differ in sample rate, the second file is resampled to match the sample rate of the first.  Handles both stereo and mono files.

# Usage
This script depends on librosa.  I recommend creating a venv in a dedicated folder to avoid risking your OS.  The window size must be specified when the code is executed.  Any value will be accepted but I strongly recommend any power of two equal to or greater than 128.  Smaller window sizes will use more RAM, especially for longer files.

`python wavmatchsynthesisstereo.py file1.wav file2.wav window-size-in-samples output.wav`
