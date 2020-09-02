# motor-imagery-deep-learning
Implementation of Convolutional Recurrent Neural Network (CRNN) to decode motor imagery EEG data.

The implementation is based on the article: [Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-based Intention Recognition for Brain Computer Interface
Dalin Zhang, Lina Yao, Xiang Zhang, Sen Wang, Weitong Chen, Robert Boots, Boualem Benatallah](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16107/0)

## Getting started
- Make sure you have Conda installed
- Create a isolated environment:
    ```
    $ conda create --name motor-imagery-deep-learning python=3.7
    ```
    
- Activate the environment:
    ```
    $ conda activate motor-imagery-deep-learning
    ``` 
- Install the dependencies:
    ```
    pip install -U -r requirements.txt
    ```

## Modules
```
/data/dataset/{dataset-name}
    description.pdf: Optional pdf with a description of the dataset
    montage.png: Optional image of the montage used to the acquisition of the EEG data
    README.md: README about the dataset and the pre-processing steps
```

```
/datasets
    {dataset-name}.py: Script to load dataset
```

```
/models/{dataset-name}
    Package of scripts with models to resolve the dataset problem.

    README.md: README with description about each model in this package
```

```
/preprocessing/{dataset-name}
    Package of scripts to perform the pre-processing steps referring to dataset
```