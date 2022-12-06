import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = load_breast_cancer()
df.head()

import sklearn.datasets
from sklearn.model_selection import train_test_split

breast_cancer = sklearn.datasets.load_breast_cancer()  # Binary classification dataset

X = breast_cancer.data
Y = breast_cancer.target

df = pd.DataFrame(breast_cancer.data,
                  columns=breast_cancer.feature_names)  # converting dataset into pandas dataframe for preprocessing
df['class'] = breast_cancer.target
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if
            df[col].dtype in [int, float]]  # eğer değişken tipi int veya float ise bunları seç

corr = df[num_cols].corr()

# koralasyon değişkenlerin birbiri ile ilişkisini ifade eden istatiksel bir ölçümdür -1 İLE 1 ARASINDA DEĞERLER ALIR

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

####### yüksek korelasyonlu değişkenlerin silinmesi

cor_matrix = df.corr().abs()  # birbirini tekrar eden değişkenler var gereksiz değişkenleri çıkartmanmız gerek

# gereksiz değişkenleri çıkartmak için :

upper_triangle_matrix = cor_matrix.where(np.ones(cor_matrix.shape), k=1).astype(np.bool)

drop_lit = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]  # 0.90 dan buyuk olanları seçip sil

cor_matrix[drop_lit]    #yüksek korelasyonlu olanlar

df.drop(drop_lit, axis=1)
