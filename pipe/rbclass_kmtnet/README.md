# GWUniverse: CNN-Based Classifier for Real and Bogus Trasient on KMTNet Dataset

**GWUniverse** is a project leveraging Convolutional Neural Networks (CNNs) to identify transient phenomena in celestial images. 
At the core of GWUniverse is a robust CNN model trained to distinguish between real transient events and bogus artifacts within the vast dataset provided by the [*Korea Microlensing Telescope Network (KMTNet)*](https://kmtnet.kasi.re.kr/kmtnet-eng/). 

Our model has been trained and fine-tuned on the KMTNet dataset, which comprises a diverse range of celestial images capturing various stages and types of transient phenomena. 
By leveraging the power of CNNs, our model achieves high accuracy and explainability in identifying genuine astronomical events, thus contributing to our understanding of the universe and supporting the astronomical community in their ongoing research endeavors.

## 1. Installation
> ***To do*** : [240321]report.md issue 01
> 1. 가상환경 테스트
> 2. 최적화된 `requirements.txt` 생성

### 1.1 Setting Up the Environment

1. Create a new virtual environment and activate it. Ensure you have `python >= 3.6`
  ```bash
  python3 -m venv gwuniverse
  ```
2. Activate the virtual environment
  - On Linux or MacOS, use:
    ```bash
    source gwuniverse/bin/activate
    ```
  - On Windows, use:
    ```bash
    .\gwuniverse\Scripts\activate
    ```  
### 1.2 Installing Dependencies
Once your environment is set up, you can install all necessary dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```

Please refer to the list below for dependency. 

```bash
albumentations
astropy
dotmap
hydra
matplotlib
numpy
omegaconf
pandas
pytorch
scikit-learn
wnadb
```
**Note** : For `pytorch`, depending on the OS you are using and whether you're using a GPU, you may need to install it manually. 
Please visit the [PyTorch official installation guide](https://pytorch.org/get-started/locally/) to find the command that fits your setup.



## 2. Usage
> To do : 
> 1. directory 규칙 설명
> 2. meta table 규칙 설명
> 3. config 사용법 설명
> 4. training 방법 설명
> 5. inference 사용방법 설명

### 2.1 Dataset config

### 2.2 Model config

### 2.3 Training

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HYDRA_FULL_ERROR=1 python train.py gpu=1 'exp_name=NGI-v0.2.95' dataset=new_dataset dataset.meta_path=/home/postech/projects/kmtnet/data/meta_near_galaxy_injectionv.0.2.95.csv
~~~

### 2.4 Inference

## 3. Datasets
> To do : 
> - KMTNet description과 reference 추가

## 4. Models
> To do : 
> - Otrain description과 reference 추가

## To do
- [ ] config.dataset에 train, test 구분

## Inference
To do:
  디렉토리 규칙 설명
  메타 테이블 규칙 설명

## Training
To do:
  훈련 방법 설명
