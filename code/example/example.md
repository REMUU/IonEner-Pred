# How to Use Models to Reproduce or Implement on Other Datasets

## Data Collection and Curation
---

### **Obtain Structures and Labels**
Structures and labels of chemicals used in this study are available in CSV format under the [code/datasets](https://github.com/REMUU/IonEner-Pred/tree/main/datasets) folder with calculated descriptors.

For researchers that would like to download the data from the NIST webbook, scripts in [code/scrape_webpage](https://github.com/REMUU/IonEner-Pred/tree/main/scrape_webpage) should be a good reference. The scraper is not for general purposes. If the researcher would like to download data from any other source, the researcher should modify the code with respect to the structure of the target website. 

In terms of missing structures, code in [code/structure_conversion](https://github.com/REMUU/IonEner-Pred/tree/main/structure_conversion) should be used to resolve SMILES structures from their name or InChI.

&nbsp;

### **Curation of Data**
Depending on the dataset, manual manipulation via Excel or Python script can be used to curate the data. Typical non-numerical data in the dataset should be modified accordingly:
| Type    | Modification                |
| ---     | ---                         |
| NaN     | Remove the descriptor column|
| Empty   | Remove the descriptor column|
| Boolean | True to 1 and False to 0    |
&nbsp;

## Descriptors Generation
---
To obtain descriptors, the [Mordred](https://github.com/mordred-descriptor) and [PaDEL](http://www.yapcwsoft.com/dd/padeldescriptor/) software should be used according to the setting described in ESI 4.1 section.

### **Example Format of the Data**
Use the IE dataset as an example:
| CAS Name | InChI | CAS Link | smiles | IE | BinaryRadical | ABC | ABCGG | ... | ... |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| Benzaldehyde, 4-(dimethylamino)- | InChI=1S/C9H11NO/c1-10(2)9-5-3-8(7-11)4-6-9/h3-7H,1-2H3 | C100107 | CN(C)c1ccc(C=O)cc1 | 7.3 | 0 | 7.956514078 | 7.451864456 | ... | ... |
| 1,4-Pentadien-3-yl radical | InChI=1S/C5H7/c1-3-5-4-2/h3-5H,1-2H2 | C14362084 | C=C[CH]C=C | 7.25 | 1 | 2.828427125 | 3.14626437 | ... | ... | 
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
&nbsp;

## Implementing the SVR Model via Scikit-Learn
---
### **Details of Model Parameters**
| Parameter | Value |
| ---       | ---   |
| gamma     | scale |
| C         | 20    |
| epsilon   | 0.01  |
&nbsp;

### **How to 10-Fold CV**
```python
# sklearn
from sklearn.preprocessing import quantile_transform
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm

# Matplotlib 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# data structure and math
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math

# define RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# define the SVR model
svr = svm.SVR(gamma='scale',
              C=20,
              epsilon=0.01)

# define a function to get model's output efficiently
def ml_model(splited_data,regressor_object):
    X_train, X_test, y_train, y_test = \
        splited_data[0], splited_data[1], splited_data[2], splited_data[3]
    regressor_object.fit(X_train, y_train)
    # get predictions from the train set
    train_prediction = regressor_object.predict(X_train)
    R2_train = np.square(pearsonr(y_train, train_prediction)[0])
    RMSE_train = rmse(y_train, train_prediction)
    # get the prediction from the test set
    test_prediction = regressor_object.predict(X_test)
    R2_test = np.square(pearsonr(y_test, test_prediction)[0])
    RMSE_test = rmse(y_test, test_prediction)
    # generate formatted output in the list
    return [R2_train, RMSE_train, R2_test, RMSE_test, test_prediction]

# io, read input csv and output in the excel
data = pd.read_csv('input_csv_path')
outdir = 'output_path'
if os.path.exists(outdir) is False:
    os.mkdir(outdir)

excel_dir = outdir + 'output_outliers_xlsx_path'
writer = pd.ExcelWriter(excel_dir)

# define the scatter plot function
def plot_scatter(predict, experiment, regressor,line_split, property, outdir):
    num_list = []
    num_list.extend(predict)
    num_list.extend(experiment)
    mini=math.floor(min(num_list))
    maxi=math.ceil(max(num_list))
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    plt.scatter(experiment,predict,s=20,c='black')
    line = mlines.Line2D([mini,maxi], [mini, maxi], color='red')
    line1 = mlines.Line2D([mini,maxi - line_split], [mini + line_split, maxi], color='red')
    line2 = mlines.Line2D([mini + line_split,maxi], [mini, maxi - line_split], color='red')
    ax.add_line(line)
    ax.add_line(line1)
    ax.add_line(line2)
    plt.title("Experimental by Predicted {} using ".format(property) + regressor + \
               '\nin 10 Fold Validation')
    plt.xlabel("Experimental {}".format(property))
    plt.ylabel("Predicted {}".format(property))
    plt.xlim(mini,maxi)
    plt.ylim(mini,maxi)
    plt.tight_layout()
    outdir = outdir + '/{}.png'.format(regressor)
    plt.savefig(outdir, dpi=500)
    plt.show()

# scale the data by quantile_transform
X = quantile_transform(data.iloc[:,5:])
y = data['IE']
kf = KFold(n_splits=10)
kf.get_n_splits(X)

# store metrics during each fold
train_cv_df = pd.DataFrame(columns=['Regressor', 'R^2', 'RMSE', 'Fold Number'])
test_cv_df = pd.DataFrame(columns=['Regressor', 'R^2', 'RMSE', 'Fold Number'])

# run the 10-Fold-CV for SVR
fold_number = 0
y_predicted_list = []
y_experimental_list = []

# store all the prediction in this DataFrame
outliers = pd.DataFrame(columns=['CAS Name','InChI','CAS Link','smiles','IE', 'Predicted IE'])

# loop for CVs of sklearn and xgb objects
for train_index, test_index in kf.split(X):
    
    fold_number += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    splited_data = [X_train, X_test, y_train, y_test]

    metrics = ml_model(splited_data, svr)
    # metrics format:
    #   [R2_train, RMSE_train, R2_test, RMSE_test, test_prediction]
    # write train metrics
    train_new_row = pd.DataFrame([['SVR', metrics[0], metrics[1], fold_number]], columns= \
        ['Regressor', 'R^2', 'RMSE', 'Fold Number'])
    train_cv_df = train_cv_df.append(train_new_row, ignore_index=True)

    # write test metrics
    test_new_row = pd.DataFrame([['SVR', metrics[2], metrics[3], fold_number]], columns= \
        ['Regressor', 'R^2', 'RMSE', 'Fold Number'])
    test_cv_df = test_cv_df.append(test_new_row, ignore_index=True)

    y_predicted_list.extend(metrics[4])
    y_experimental_list.extend(y_test)
    print('SVR', 'at fold ', fold_number)
    
    for i in range(len(test_index)):
        ind_name = test_index[i]
        mol = data.iloc[:,:5].iloc[[ind_name]].values[0]
        new_row = pd.DataFrame([[mol[0], mol[1], mol[2], mol[3], mol[4], metrics[4][i]]], \
            columns=['CAS Name','InChI','CAS Link','smiles','IE', 'Predicted IE'])
        outliers = outliers.append(new_row, ignore_index=True)

# store all prediction results in the excel 
ots = outliers
ots.to_excel(writer, sheet_name=algorithm)
# draw scatter plot of the prediction
plot_scatter(y_predicted_list, y_experimental_list, algorithm, 1, 'IE',outdir)

# output section of 
writer.save()

train_dir = outdir + 'output_train_excel_path'
train_cv_df.to_excel(train_dir)
train_cv_df

test_dir = outdir + 'output_test_excel_path'
test_cv_df.to_excel(test_dir)
test_cv_df
```

&nbsp;

## Implementing the AttentiveFP via DGL-LifeSci
---
### **Details of Model Parameters**
| Parameter       | Value  |
| ---             | ---    |
| graph_feat_size | 200    |
| num_layers      | 2      |
| batch_size      | 128    |
| patience        | 30     |
| num_timesteps   | 2      |
| learning rate   | 0.0003 |
&nbsp;

### **How to 10-Fold CV**

Example command line input:

**Training:**

```
$ python regression_train.py -c input.csv -mo AttentiveFP -sc smiles -t IE -s random -sr 0.7,0.15,0.15 -me rmse -a canonical -b canonical -n 300 -nw 8 -pe 10 -p output_path
```

**Optimization:**

```
$ python regression_train.py -c input.csv -mo AttentiveFP -sc smiles -t IE -s random -sr 0.7,0.1,0.2 -ne 200 -me rmse -a canonical -b canonical -n 1000 -nw 8 -pe 10 -p output_path
```

**10-Fold-Cross Validation:**

The [modified_regression_train.py](https://github.com/REMUU/IonEner-Pred/blob/main/code/gnn/modified_regression_train.py) is use to run the 10-Fold-CV as the need of this research.

```
$ python modified_regression_train.py -c input.csv -mo AttentiveFP -sc smiles -t IE -s consecutive_smiles -me rmse -a canonical -b canonical -n 300 -nw 8 -pe 10 -p output_path
```

**Read Results of 10-Fold-Cross Validation:**

The following script is used to obtain the data for 10-Fold-CV. Please implement it with your own modification to specific cases.

```python
import pandas as pd
import os
from scipy.stats import pearsonr
import numpy as np

# define function for RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# io
path = os.getcwd()
out_excel_path = path + '/metrics.xlsx'

# metrics will be stored in the following format
metrics_df = pd.DataFrame(columns=['Algorithm','Fold No.','Test R2','Test RMSE'])

# read prediction results from 10 fold in the folder (for instance AttentiveFP_10fold)
for cv in ['AttentiveFP', 'GAT', 'MPNN', 'NF', 'GCN', 'Weave']:
    cv_df = pd.DataFrame(columns=['SMILES','LABELS','PREDICTIONS'])
    cv_path = path + '/' + cv + '_10fold'
    # screen metrics for 10 folds
    for i in range(10):
        k = i+1
        test_path = cv_path + '/' + str(k) + '/test_output.csv'
        test_df = pd.read_csv(test_path)
        test_metrics = pd.DataFrame([[cv, k, \
            np.square(pearsonr(test_df['LABELS'], \
            test_df['PREDICTIONS'])[0]), \
            rmse(test_df['PREDICTIONS'], \
            test_df['LABELS'])]], \
            columns=['Algorithm','Fold No.','Test R2','Test RMSE'])
        metrics_df = metrics_df.append(test_metrics,ignore_index=True)
        cv_df = cv_df.append(test_df)
    out_csv_path = path + '/' + cv + '.csv'
    cv_df.to_csv(out_csv_path, index=False)

# write metrics in the excel
metrics_df.to_excel(out_excel_path, index=False)
```

&nbsp;

## Notice
---

## **Application Domain**
This study is not intended to bring a model to solve all problems. However, it should generate satisfying resutls under properly prepared dataset. 

### **Following elements are found during our training of models:**

```
[ H,
  B, C, N, O, F, 
  Si, P, S, Cl, 
  Ge, As, Se, Br, 
  I ] 
```

It is relatively safe to apply the model with organic chemicals containing no more than these elements. We recommend using SMILES as the chemical representation to calculate descriptors. We do not guarantee the result will be satisfying when applying the model to larger organic molecules, polymers, proteins, any inorganic compounds, or periodic systems. Please use the model with caution.