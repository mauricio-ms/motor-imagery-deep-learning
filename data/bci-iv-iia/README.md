# BCI Competition IV Dataset IIA
[BCI Competition IV](http://www.bbci.de/competition/iv)

Each EEG record has one of the following codes:
    
    276   0x0114 Idling EEG (eyes open)
    277   0x0115 Idling EEG (eyes closed)
    768   0x0300 Start of a trial
    769   0x0301 Cue onset left (class 1)
    770   0x0302 Cue onset right (class 2)
    771   0x0303 Cue onset foot (class 3)
    772   0x0304 Cue onset tongue (class 4)
    783   0x030F Cue unknown
    1023  0x03FF Rejected trial
    1072  0x0430 Eye movements
    32766 0x7FFE Start of a new run
    
According with these rules, each EEG record in the gdf files will be mapped to one of the following events in the csv files:

    0: LEFT_HAND
    1: RIGHT_HAND
    2: FOOT
    3: TONGUE