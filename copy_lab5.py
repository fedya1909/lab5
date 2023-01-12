from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import cluster
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True

# Представим, что предупреждения нас не касаются
warnings.filterwarnings("ignore")


def num_of_labels(model):
  return len(pd.Categorical(model.labels_).unique())


# Загружаем данные
df = pd.read_csv("SMSSpamCollection", error_bad_lines = False, sep='\t')
print(df.head(10))
print()
