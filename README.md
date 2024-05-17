# wavmatchsynthesis
This script slices two .wav files into windows and reconstructs the first file by matching its windows with those from the second. Windows are modified by a Hanning function with 50% overlap. If two files differ in sample rate, the second file is resampled to match the sample rate of the first.

# Usage
`python wavmatchsynthesisstereo.py file1.wav file2.wav window-size-in-samples output-file`
