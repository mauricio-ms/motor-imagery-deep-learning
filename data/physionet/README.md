# Physionet Dataset
[EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0)

Each EEG record has one of three codes (T0, T1, or T2):
    
    T0 corresponds to rest
    
    T1 corresponds to onset of motion (real or imagined) of:
        the left fist (in runs 3, 4, 7, 8, 11, and 12)
        both fists (in runs 5, 6, 9, 10, 13, and 14)
    
    T2 corresponds to onset of motion (real or imagined) of:
        the right fist (in runs 3, 4, 7, 8, 11, and 12)
        both feet (in runs 5, 6, 9, 10, 13, and 14)

According with these rules, each EEG record in the edf-files will be mapped to one of the following events in the csv files:

    "0": "REST"
    "1": "LEFT_FIST"
    "2": "RIGHT_FIST"
    "3": "BOTH_FISTS"
    "4": "BOTH_FEET"