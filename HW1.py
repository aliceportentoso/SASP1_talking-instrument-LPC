# DAAP Homework 1
# students and students code:
# Chiara Lunghi     233195
# Alice Portentoso  232985

import numpy as np
import wave
from scipy.linalg import toeplitz
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

# Load the audio file and generate a NumPy array that contains its normalized samples
def load_signal(filename):

    with wave.open(filename, 'rb') as f:

        # Get audio parameters
        global num_sample, rate
        num_sample = f.getparams().nframes
        rate = f.getparams().framerate

        # Read frames and convert into array
        frames = f.readframes(num_sample)
        signal = np.frombuffer(frames, dtype=np.int16)

        # Normalize the signal
        signal = signal / np.float32(32767)

    return signal

# Divide the signal into frames and windowing
def divide_into_frames_and_window(signal):

    # Calculate sizes
    global step_size
    global num_frames
    step_size = int(frame_size * (1 - overlap_factor))          # 512
    num_frames = (num_sample - frame_size) // step_size + 1     # 1657
    frames = np.zeros((num_frames, frame_size))

    # Divide into frames and windowing using hanning function
    for i in range(num_frames):
        frames[i, :] = signal[i * step_size: i * step_size + frame_size] * np.hanning(frame_size)
    return frames

# Compute the auto-correlation matrix and the cross-correlation vector
def correlation(frame, order):

    # Calculate auto-correlation vector r, auto-correlation matrix R and cross-correlation vector P
    r = np.zeros(order)
    r = np.correlate(frame, frame, "full")[frame_size - 1:]
    P = np.zeros(order)
    P = r[1:order + 1]
    R = np.zeros((order, order))
    R = toeplitz(r[:order])
    return P, R

# Whitening filter (LPC) using closed-form solution of the Wiener-Hopf equations
def lpc_HW(frame, order):

    # Compute R and P from the input frame
    P, R = correlation(frame, order)

    # Solve Wiener-Hopf equations
    a = np.linalg.solve(R, P)

    return a

# Whitening filter (LPC) using steepest descent algorithm
def lpc_SD(frame, order):

    # compute R and P from the input signal
    P, R = correlation(frame, order)

    # choose mu in order to have convergence (stability condition)
    mu = 0.95*(2/max(np.linalg.eigvals(R)))
    iter = 100

    # Stability Control on R matrix
    if not (stab_control(mu, R)):
        print('Stability Control Failed!')

    # Initialize a vector as an empty vector
    a = np.zeros((order,))

    # Next guess at the tap-weight vector a
    for i in range(iter):
        a = a + mu*(P - np.dot(R, a))
    return a


# Check stability control for convergence
def stab_control(mu, R):

    # Find the max eighvalue of R
    eig = np.linalg.eigvals(R)
    max_eig = max(eig)

    # Verify the convergence
    if mu > 0 and mu < (2 / max_eig):
        return True
    else:
        return False

# Sum the frames together and gets the output signal
def overlap_and_add(frames, size):

    # Initialize output signal with final length
    output = np.zeros((num_frames - 1) * step_size + size)

    # Iterate on frames and sum them into output signal
    for i in range(num_frames):
        output[i * step_size: (i * step_size + size)] += frames[i, :size]

    return output

# Print COLA condition result
def cola():
    cola = np.zeros(num_sample)

    for i in range(num_frames):
        cola[i * step_size:i * step_size + frame_size] += np.hanning(frame_size)

    #Show a plot of overlapped windows
    plt.figure(figsize=(12, 6))

    plt.plot(cola)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('COLA condition')

    plt.show()

# Framing parameters
frame_size = 1024
overlap_factor = 0.5
# Filters parameters
order_piano = 24
order_speech = 48

# Load signal
piano_signal = load_signal('piano.wav')
speech_signal = load_signal('speech.wav')
print("Signal loaded and normalized")

# Divide signal into frames and windowing with hanning
piano_frames = divide_into_frames_and_window(piano_signal)
speech_frames = divide_into_frames_and_window(speech_signal)
print("Signal divided into frames and added window")

# COLA condition
cola()

# Frame size after zero padding
final_frame_size = 4 * frame_size

filtered_frames = np.zeros((num_frames, final_frame_size))

# Choose LPC analysis method: HW or SD
LPC_analysis_method = 'HW'

for i in range(num_frames):

    if LPC_analysis_method == 'HW':
        # LPC analysis of piano signal with Wiener-Hopf equations (WH):
        a_piano = lpc_HW(piano_frames[i], order_piano)
        a_speech = lpc_HW(speech_frames[i], order_speech)
    if LPC_analysis_method == 'SD':
        # LPC analysis coof piano signal using steepest descent algorithm (SD):
        a_piano = lpc_SD(piano_frames[i], order_piano)
        a_speech = lpc_SD(speech_frames[i], order_speech)

    # Create whitening filters from LPC coefficients
    w_filter_piano = np.zeros(order_piano + 1)
    w_filter_piano[0] = 1
    w_filter_piano[1:] = -a_piano
    w_filter_speech = np.zeros(order_speech + 1)
    w_filter_speech[0] = 1
    w_filter_speech[1:] = -a_speech

    # Transform signal and filters in frequency domain with zero padding
    frame_freq = np.fft.fft(piano_frames[i], n = final_frame_size)
    w_filter_piano_freq = np.fft.fft(w_filter_piano, n = final_frame_size)
    w_filter_speech_freq = np.fft.fft(w_filter_speech, n = final_frame_size)

    # Apply whitening filter of piano and shaping filter of speech to the frame and return in time domain
    s_filter_speech_freq = 1 / w_filter_speech_freq
    filtered_frame = np.fft.ifft(frame_freq * w_filter_piano_freq * s_filter_speech_freq).real

    # Add new frames to an array of filtered frames
    filtered_frames[i] = filtered_frame

# Overlap and add new frames with zero padding
output_signal = overlap_and_add(filtered_frames, final_frame_size)
print("Overlap and add done")

# Denormalize output signal
for i in range(len(output_signal)):
    output_signal[i] = output_signal[i] * np.float32(32767)
print("Denormalized output signal")

# Print and save final signal
print("Output signal: ", output_signal)
wavfile.write("output_signal.wav", rate, output_signal.astype(np.int16))
print("File audio result saved")
