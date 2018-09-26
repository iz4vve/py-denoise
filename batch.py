import glob
import os
import sys

import tqdm
import denoise


def main():
    
    params1 = {
        "--cutoff": 400,
        "--algo": "window_filter",
        "--window": "hanning",
    }

    params2 = {
        "--cutoff": 400,
        "--algo": "butterworth",
        "--window": "hanning",
        "--low": 2000,
        "--high": 18000,
        "--filter-order": 5
        # --filter-type=<filter>  Type of filter for Butterworth. Should be one of 'lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’ [default: lowpass].
    }

    files = glob.glob(sys.argv[1])
    print("{} files to convert...".format(len(files)))

    # WINDOW
    print("Window filter -> using {} window".format(params1["--window"]))
    for f in tqdm.tqdm(files):
        new_name = f.replace(".wav", "_hanning.wav")
        new_name = new_name.split("/")[-1]
        params1["--input"] = f
        params1["--output"] = new_name

        denoise.window_filter(params1)

    # BUTTERWORTH
    print("Butterworth filter")
    for t in ("lowpass", "highpass"):#, "bandpass"):
        print("Applying {} Butterworth filter".format(t))
        for f in tqdm.tqdm(files):
            new_name = f.replace(".wav", "_{}.wav".format(t))
            new_name = new_name.split("/")[-1]
            params2["--input"] = f
            params2["--output"] = new_name
            params2["--filter-type"] = t

            denoise.butter(params2)

    print("Applying bandpass Butterworth filter")
    for f in tqdm.tqdm(files):
        new_name = f.replace(".wav", "_highed.wav")
        new_name = new_name.split("/")[-1]
        params2["--input"] = f
        params2["--output"] = new_name
        params2["--filter-type"] = "highpass"
        params2["--cutoff"] = 200

        denoise.butter(params2)

        final_name = new_name.replace("highed", "bandpass")
        params2["--input"] = new_name
        params2["--output"] = final_name
        params2["--filter-type"] = "lowpass"
        params2["--cutoff"] = 3600

        denoise.butter(params2)
        os.remove(new_name)



if __name__ == "__main__":
    main()
