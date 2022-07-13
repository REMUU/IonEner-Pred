import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import preprocessing
import seaborn as sns

# define data preparation tools
df = pd.read_csv(r'input_csv_path')
X = df.iloc[:, 5:]
X1 = preprocessing.normalize(X, norm='l2')
X2 = preprocessing.power_transform(X)
X3 = preprocessing.binarize(X)
X4 = preprocessing.maxabs_scale(X)
X5 = preprocessing.minmax_scale(X)
X6 = preprocessing.quantile_transform(X)
scaler = preprocessing.StandardScaler().fit(X)
X7 = scaler.transform(X)
X_list = [X1, X2, X3, X4, X5, X6, X7]
Y = df['IE']
position = 0
# define the output excel file
write = pd.ExcelWriter(r'output_PCA_xlsx_path')

# iterate on different data preprocessing for the PCA analysis
for X in X_list:
    position += 1
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    # write excel
    pd.DataFrame(X).to_excel(write, sheet_name='X{}'.format(position))
    sns.set(style="darkgrid")
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    cm = plt.cm.get_cmap('RdYlBu')
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    sc = ax.scatter(x, y, z, c=Y, cmap=cm, s=2)
    cb = plt.colorbar(sc)
    cb.set_label('IE/eV')
    plt.tight_layout()
    # plot PCA figures
    plt.savefig(r'output_png_path'.format(position),dpi=328)
    plt.show()
write.save()
