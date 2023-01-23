## About the Paper

Title: ***Predict Ionization Energy of Molecules Using Conventional and Graph Based Machine Learning Models***

Author: **Yufeng Liu**, **Zhenyu Li***

>**Abstract**: Ionization energy (IE) is an important property of molecules. It is highly desirable to predict IE efficiently based on, for example, machine learning (ML)-powered quantitative structure-property relationships (QSPR). In this study, we systematically compare the performance of different machine learning models in predicting the IE of molecules with distinct functional groups obtained from the NIST webbook. Mordred and PaDEL are used to generate informative and computationally inexpensive descriptors for conventional ML models. Using a descriptor to indicate if the molecule is a radical can significantly improve the performance of these ML models. Support vector regression (SVR) is the best conventional ML model for IE prediction. In graph-based models, the AttentiveFP gives an even better performance compared to SVR. The difference between these two types of models mainly comes from their predictions for radical molecules, where the local environment around an unpaired electron is better described by graph-based models. These results provide not only high-performance models for IE prediction but also useful information in choosing models to obtain reliable QSPR.

<u>Any potential usage of codes and data should directly refer to this paper or their original publications.</u>

## Repository Structure
- catagorized_images
	- catagorized_images.zip
- code
	- conventional_machine_learning
	- example
	- feature_selection
	- gnn
	- nni
	- pca
	- plot
	- scrape_webpage
	- structure_conversion
- dataset 
	- freesol_by_descriptor_set
	- full_set
	- lipophilicity_by_descriptor_set
	- nist_descriptor_by_descriptor_set

## Dependencies
| Package      | Version     |
| ---          | ---         |
| Python       | 3.7.0       |
| Scikit-Learn | 0.24.2      | 
| Tensorflow   | 2.6.0       |
| dgllife      | 0.2.8       |
| DGL          | 0.7.1       |
| PyTorch      | 1.9.0       |
| RDKit        | 2018.09.3.0 |
| Mordred      | 1.2.0       |


## Citation
Yufeng Liu and Zhenyu Li*, Journal of Chemical Information and Modeling Article, ASAP, DOI: 10.1021/acs.jcim.2c01321