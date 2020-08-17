# motor-imagery-convolutional-recurrent-neural-network
Implementation of Convolutional Recurrent Neural Network (CRNN) to decode motor imagery EEG data.

The implementation is based on the article: [Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-based Intention Recognition for Brain Computer Interface
Dalin Zhang, Lina Yao, Xiang Zhang, Sen Wang, Weitong Chen, Robert Boots, Boualem Benatallah](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16107/0)

## Getting started
- Make sure you have Conda installed
- Creating a isolated environment:

    $ conda create --name motor-imagery-convolutional-recurrent-neural-network python=3.7
    
- Activating the environment:

    $ conda activate motor-imagery-convolutional-recurrent-neural-network
    
- Installing the dependencies:

    $ pip install -U -r requirements.txt

## Modules
`/data`
```
/dataset
    description.pdf: Contains a description of the dataset
    labels.json: Contains a json structure to documents the mapping between the class values and a representative description
    original-events-mapping.m: Contains a description to documents the mapping between the original event values and the respective classes
```

`/preprocessing`
```
- Conversion of eeg original data files to csv files;
- Normalization of eeg data.
```

## Models
Model: "cnn_1d"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 64, 32)            128       
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 64, 64)            6208      
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 64, 128)           24704     
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0         
_________________________________________________________________
dropout (Dropout)            (None, 8192)              0         
_________________________________________________________________
dense (Dense)                (None, 1024)              8389632   
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 5125      
=================================================================
Total params: 8,425,797
Trainable params: 8,425,797
Non-trainable params: 0
_________________________________________________________________

# TO DO
- Update folders documentation in readme
- Add validation in `verify_integrity_raw_csv_files_physionet` to verify if files has samples skipped by the label `-1`