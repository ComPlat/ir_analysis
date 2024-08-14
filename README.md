# Infrared Spectrum analysis

Publication link : 

Automated Infrared spectrum analysis to identify 17 functional groups using deep learning. 
At present our model can identify below mentioned groups. 
Our model can be extended to incorporate identification of other functional groups as well.

    - alkane
    - methyl
    - alkene
    - alkyne
    - alcohols
    - amines
    - nitriles
    - aromatics
    - alkyl halides
    - esters
    - ketones
    - aldehydes
    - carboxylic acids
    - ether
    - acyl halides
    - amides
    - nitro

Our data pipeline processes Jcamp files and structures the data into dataframes.
![Alt text](./data_pipeline.png)

Our published models were trained on NIST SRD 35 and Chemotion datasets.

We cannot provide NIST SRD 35 dataset as it is a commericial dataset. 

We provide our open access Chemotion dataset which can be downloaded via Radar4Chem : 
https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst


## Table of Contents

- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/devpunjabi/ir_analysis_kit.git
    ```

2. Change to the project directory:

    ```sh
    cd ir_analysis_kit
    ```

3. Create and activate an anaconda virtual environment:

    ```sh
    conda create --name ir python=3.8
    conda activate ir

    ```

4. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to data folder. 
```
    cd ir_analysis/data
```
2. Follow the instructions in data_preproc.ipynb to create preprocessed datasets.

3. Export dataset file as environment variable 
```
    export DF=~/data/preprocessed_dataset.pk
```
4. Navigate to utils folder 
```
    cd ir_analysis/utils
```
5. Update config.json with desired parameters. Default parameters in the file belong to our best model. 

6. Run K-fold cross validation with 
```
    python cv.py
```
7. Alternative : train and test for 1 fold of the dataset
```
    python train_test.py --config config.json --results
```
8. Trained models are saved under models folder with experiment names mentioned in config.json

## Contributing

Contributions are welcome! Please follow these steps to contribute:

    -Fork the repository.

    -Create a new branch: git checkout -b feature-branch-name.

    -Make your changes and commit them: git commit -m 'New feature'.

    -Push to the branch: git push origin feature-branch-name.

    -Open a pull request.


## License

This project is licensed under the MIT License


## Contact

Dev Punjabi - dev.punjabi@kit.edu,

Nicole Jung - nicole.jung@kit.edu

Institution : https://www.ibcs.kit.edu/