"""
The module denoise.py contains functions and utilities to perform denoising
operations on a wav file.

Operations implemented:
    - apodization;

Planned:
    - Butterwoth filter;
    - Low-pass filter;
    - High-pass filter;
    - Band-pass fileter;
    - ...


Licensed under the MIT License;
you may not use this file except in compliance with the License.
Copy of the License is included in this repository

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import wave

import numpy as np
from scipy import signal

__author__ = "Pietro Mascolo"

WINDOWS = {
    "barthann": signal.barthann,
    "bartlett": signal.bartlett,
    "blackman": signal.blackman,
    "boxcar": signal.boxcar,
    "cosine": signal.cosine,
    "exponential": signal.exponential,
    "hann": signal.hann,
    "hamming": signal.hamming,
    "hanning": signal.hanning,
}


def window_filter(arguments: dict):
    """
    window_filter convolves a window filter with a signal to denoise it
    """
    _in, _out = arguments["--input"], arguments["--output"]
    cutoff = int(arguments["--cutoff"])
    metadata = get_metadata(arguments["--input"])

    # cut off frequency for the filter
    freq_ratio = (cutoff / metadata["sample_rate"])
    window_size = int(np.sqrt(0.196196 + freq_ratio ** 2) / freq_ratio)

    filtered = apply_window(
        metadata["channels"][0],
        window_size,
        dtype=metadata["dtype"],
        window=arguments["--window"]
    )

    with wave.open(_out, "w") as output_audio:
        output_audio.setparams(
            (
                1,
                metadata["sample_width"],
                metadata["sample_rate"],
                metadata["n_frames"],
                metadata["comp_type"],
                metadata["comp_name"]
            )
        )
        output_audio.writeframes(filtered.tobytes('C'))
        output_audio.close()


def apply_window(x, windowSize, dtype=np.int16, window="hamming"):
    """
    apply_window convolves a window shape to a signal 
    to achieve apodization and noise reduction
    """
    win = WINDOWS.get(window, "hamming")(windowSize)
    return signal.lfilter(win, [1], x).astype(dtype)


def get_metadata(audio_path: str, interleaved=True) -> dict:
    """
    get_channel returns channel information for a specific wav file
    """
    with wave.open(audio_path, "rb") as audio:
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        n_channels = audio.getnchannels()
        n_frames = audio.getnframes()
        data = audio.readframes(n_frames * n_channels)
        comp_type = audio.getcomptype()
        comp_name = audio.getcompname()

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only 8 and 16 bit audio formats are supported.")

    channels = np.fromstring(data, dtype=dtype)

    if interleaved:
        # sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return {
        "channels": channels,
        "sample_width": sample_width, 
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "n_frames": n_frames,
        "signal": data,
        "comp_type": comp_type,
        "comp_name": comp_name,
        "dtype": dtype
    }


def butter(arguments: dict):
    """
    butter applies a Butterworth filter to a signal
    """
    _in, _out = arguments["--input"], arguments["--output"]
    low, high = float(arguments["--low"]), float(arguments["--high"])
    cutoff = float(arguments["--cutoff"])
    filter_type = arguments["--filter-type"]
    order = float(arguments["--filter-order"])
    metadata = get_metadata(arguments["--input"])

    nyquist = metadata["sample_rate"] * 0.5

    if filter_type in ("lowpass", "highpass"):
        b, a = signal.butter(order, cutoff / nyquist, btype=filter_type)
    elif filter_type in ("bandpass", "bandstop"):
        Wn = [low / nyquist, high / nyquist]
        b, a = signal.butter(order, Wn, btype=filter_type)

    filtered = signal.lfilter(b, a, metadata["channels"][0]).astype(metadata["dtype"])

    with wave.open(_out, "w") as output_audio:
        output_audio.setparams(
            (
                1,
                metadata["sample_width"],
                metadata["sample_rate"],
                metadata["n_frames"],
                metadata["comp_type"],
                metadata["comp_name"]
            )
        )
        output_audio.writeframes(filtered.tobytes('C'))
        output_audio.close()