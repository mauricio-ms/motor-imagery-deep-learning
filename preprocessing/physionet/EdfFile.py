import re
import pyedflib
import os
import numpy as np


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
        self.data = self.__read()

    def get_path_file(self, path_dir, extension):
        return os.path.join(path_dir, f"S{self.subject}R{self.run_execution}.{extension}")

    def __read(self):
        data = np.zeros((self.n_samples, self.n_channels + 1))
        # Set invalid label to verify skipped samples
        data[:, -1] = -1
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
            data[onset_index:end_index, -1] = np.repeat(self.__get_label_for_event(event), event_samples)

        return data

    def __get_label_for_event(self, event):
        if event == "T0":
            return 0

        movement_single_member_runs = [3, 4, 7, 8, 11, 12]
        movement_both_members_runs = [5, 6, 9, 10, 13, 14]

        if event == "T1":
            if self.run_execution in movement_single_member_runs:
                return 1
            elif self.run_execution in movement_both_members_runs:
                return 3

        if event == "T2":
            if self.run_execution in movement_single_member_runs:
                return 2
            elif self.run_execution in movement_both_members_runs:
                return 4
