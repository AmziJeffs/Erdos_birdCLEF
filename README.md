# Bird Species identification on audio samples using CNNs (BirdCLEF 2024)


## Authors
[Amzi Jeffs](https://github.com/AmziJeffs)  
[Junichi Koganemaru](https://github.com/jkoganem)  
[Salil Singh](https://github.com/sllsnghlrns)  
[Ashwin T.A.N.](https://github.com/ashwintan1)     

## Overview

Climate change plays a devastating role in the destruction of natural habitats and the global decline of biodiversity. One way of understanding these effects is by monitoring bird populations as they are highly migratory and have diverse habitual needs. However, traditional methods can be costly and logistically challenging to conduct at large scales. 

In this project we explore using passive acoustic monitoring (PAM) combined with deep learning methods to identify bird specifies using audio samples in the Western Ghats of India. The broader goal of this project is to contribute to the growing literature of deep learning approaches to tackling issues brought by climate change. 

## Structure of repository

- `all scripts` folder contains all possible 
- `data` folder contains the training metadata. .py scripts in the scripts folders can be ran locally after populating the `data` folder with the official training audio data. 
- `sample scripts` folder contains scripts used for exploratory data analysis (EDA), a baseline 2 layer CNN model and also an improved 6 layer CNN model with improved data augmentation. 

## Description of dataset

We use the dataset provided by the [BirdCLEF 2024 competition](https://www.kaggle.com/competitions/birdclef-2024), hosted on Kaggle. We are asked to train a model reporting probabilities of the presence of 182 given bird species in hidden test set of ~1100 audio clips; each test clip contains potentially multiple bird calls. 

The metadata for the training audio samples can be found under the [data folder](data/test_metadata.csv). The rest of the data are available on [the official competition webpage](https://www.kaggle.com/competitions/birdclef-2024/data).   
 

## Evaluation metric

The model is evaluated using a macro-averaged ROC-AUC metric on a hidden testing set provided by the organizers; see [the official documentation](https://www.kaggle.com/competitions/birdclef-2024/overview/evaluation) for more details. 

## Exploratory data analysis


## Model architecture 

We implemented a ... 







## Acknowledgement 

We would like to thank the organizers and the associated organizations of the BirdCLEF 2024 competition for hosting the competition. We would like to thank the Erdös institute for providing the authors the opportunity to work on this project as part of the Erdös institute Data Science Bootcamp. We would also like to thank Nuno Chagas and the Department of Mathematical Sciences at Carnegie Mellon University for providing computing support for the project. 

## References 