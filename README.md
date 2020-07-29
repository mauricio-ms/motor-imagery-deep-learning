# motor-imagery-convolutional-recurrent-neural-network
Implementation of Convolutional Recurrent Neural Network (CRNN) to decode motor imagery EEG data.

The implementation is based on the article: [Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-based Intention Recognition for Brain Computer Interface
Dalin Zhang, Lina Yao, Xiang Zhang, Sen Wang, Weitong Chen, Robert Boots, Boualem Benatallah](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16107/0)

## Getting started
- Make sure you have Conda installed
- Creating a isolated environment:

    $ conda create --name motor-imagery-crnn python=3.7
    
- Activating the environment:

    $ conda activate motor-imagery-crnn
    
- Installing the dependencies:

    $ pip install -U -r requirements.txt

## Modules
`/preprocessing`
* Conversion of eeg original data files to csv files;
* Normalization of eeg data. 