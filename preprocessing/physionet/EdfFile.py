import re
import pyedflib
import os
import numpy as np

MOVEMENT_SINGLE_MEMBER_RUNS = [3, 4, 7, 8, 11, 12]
MOVEMENT_BOTH_MEMBERS_RUNS = [5, 6, 9, 10, 13, 14]


class EdfFile:
    def __init__(self, edf_path_dir, edf_file_name):
        self.edf_path_dir = edf_path_dir
        self.edf_file_name = edf_file_name

        groups_edf_file = re.match("S(\\d+)R(\\d+).edf", edf_file_name).groups()
        self.subject = groups_edf_file[0]
        self.run_execution = int(groups_edf_file[1])
        self.__file = pyedflib.EdfReader(os.path.join(edf_path_dir, edf_file_name))
        self.channels_labels = self.__file.getSignalLabels()

        annotations = self.__file.readAnnotations()
        self.__onset_events = annotations[0]
        self.__duration_events = annotations[1]
        self.__events = annotations[2]

        self.frequency = self.__file.getSampleFrequencies()[0]
        self.n_samples = int(np.round(np.sum(self.__duration_events), decimals=2) * self.frequency)
        self.n_channels = self.__file.signals_in_file
        self.data, self.labels = self.__read()

    def get_path_file(self, path_dir, extension):
        return os.path.join(path_dir, f"S{self.subject}R{self.run_execution}.{extension}")

    def __read(self):
        data = np.zeros((self.n_samples, self.n_channels))
        # Set invalid label to verify skipped samples
        labels = np.full(self.n_samples, "invalid", dtype="U256")
        end_index = None
        for index_event in range(len(self.__onset_events)):
            onset_event = self.__onset_events[index_event]
            duration_event = self.__duration_events[index_event]
            event = self.__events[index_event]

            onset_index = int(onset_event * self.frequency)
            if end_index is not None and onset_index != end_index:
                onset_index = end_index
            end_index = np.minimum(int(np.round(onset_event + duration_event, decimals=2) * self.frequency),
                                   self.n_samples)
            event_samples = end_index - onset_index

            for ch in np.arange(self.n_channels):
                data[onset_index:end_index, ch] = self.__file.readSignal(ch, onset_event, event_samples)
            labels[onset_index:end_index] = np.repeat(self.__get_label_for_event(event), event_samples)

        if np.sum(labels == "invalid") > 0:
            print("WARNING: Samples skipped when reading the file " + self.edf_file_name)

        return data, labels

    """
        The events of real movements are not handled because these files are not read
    """
    def __get_label_for_event(self, event):
        if event == "T0":
            if self.run_execution == 1:
                return "eyes-open"
            elif self.run_execution == 2:
                return "eyes-closed"
            else:
                return "rest"

        if event == "T1":
            if self.run_execution in MOVEMENT_SINGLE_MEMBER_RUNS:
                return "left-fist"
            elif self.run_execution in MOVEMENT_BOTH_MEMBERS_RUNS:
                return "both-fists"

        if event == "T2":
            if self.run_execution in MOVEMENT_SINGLE_MEMBER_RUNS:
                return "right-fist"
            elif self.run_execution in MOVEMENT_BOTH_MEMBERS_RUNS:
                return "both-feet"

        return None
