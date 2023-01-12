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
df = pd.read_csv("dow_jones_index.data")

print(df.head(10))
print()
cols = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
for col in cols:
  df[col] = df[col].str.replace('$','')
df = pd.get_dummies(df,columns=['stock'])
df = df.drop(columns=['date'])
"""df['date'] = df['date'].apply(
    lambda x: datetime.strptime(x, '%m/%d/%Y'))
df['day'] = df['date'].dt.day
df['month']=df['date'].dt.month
df['year']=df['date'].dt.year
df = df.drop(columns=['date'])
"""
df['percent_change_volume_over_last_wk'] = df['percent_change_volume_over_last_wk'].fillna(
  df['percent_change_volume_over_last_wk'].median())
df['previous_weeks_volume'] = df['previous_weeks_volume'].fillna(
  df['previous_weeks_volume'].median())

cols = df.select_dtypes(exclude=['float64']).columns
df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
df['volume'] = df['volume'].astype(int)
print(df.dtypes)
print(df.isnull().sum())
print(df.head())
print(df.info())
print(df.describe().round(2))
#Масштабирование всего массива до приблизительных значений дисперсии, чтобы улучшить точность результатов
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled,columns=df.columns)
print(df_scaled.head())

pca = PCA(2)
df_centered = df_scaled - df_scaled.mean(axis=0)
pca.fit(df_centered)
pca_cenered = pca.transform(df_centered)


def plot(labels=None):
  flat = PCA(3).fit_transform(df_centered)
  plt.scatter(flat[:, 0], flat[:, 1], s=1, c='blue' if labels is None else pd.Categorical(
      labels).codes, cmap='Set3')
  plt.show()
plot()

#Кластеризация K-Means
for_plot = {x: [] for x in ["clusters", "silhouette", "calinski_harabasz"]}
for n_clusters in range(2, 15):
  kmeans = cluster.KMeans(n_clusters).fit(df)

  for_plot['clusters'].append(n_clusters)
  for_plot['calinski_harabasz'].append(
      metrics.calinski_harabasz_score(df, kmeans.labels_))
  for_plot['silhouette'].append(metrics.silhouette_score(df, kmeans.labels_))

fig, axs = plt.subplots(2, 1)
for ax, score in zip(axs, ['silhouette', 'calinski_harabasz']):
  ax.plot(for_plot['clusters'], for_plot[score], label=score)
  ax.legend()

plt.xlabel("Количество кластеров")
plt.show()

# С KMean лучшее число кластеров=3
plot(cluster.KMeans(3).fit(df).labels_)

# Пробуем алгоритм MeanShift. 
ms = cluster.MeanShift().fit(df)

print(
    f"calinski_harabasz_score={metrics.calinski_harabasz_score(df, ms.labels_)}")
print(f"silhouette_score={metrics.silhouette_score(df, ms.labels_)}")
print(f"Количество кластеров={num_of_labels(ms)}")
plot(ms.labels_)

# Пробуем DBSCAN
'''
for_plot = {x: []
            for x in ["eps", "min_samples", "silhouette", "calinski_harabasz"]}
for eps_step in range(1, 8, 4):
  eps = 6
  dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(df)

  for_plot['eps'].append(eps)
  for_plot['min_samples'].append(min_samples)
  for_plot['calinski_harabasz'].append(metrics.calinski_harabasz_score(df, dbscan.labels_))
  for_plot['silhouette'].append(metrics.silhouette_score(df, dbscan.labels_))
for_plot = pd.DataFrame(for_plot)

by_eps = for_plot.groupby(by="eps")
fig, axs = plt.subplots(2, len(by_eps), sharey='row',
                        sharex='col', figsize=(10, 6))
for idx, group in enumerate(by_eps):
  eps, d = group
  axs[0, idx].set_title(f"Eps={eps}")
  axs[0, idx].plot(d['min_samples'], d["silhouette"], label="silhouette")
  axs[1, idx].plot(d['min_samples'], d["calinski_harabasz"], label="calinski_harabasz", color='red')

  if idx == 3:
    axs[0, idx].legend()
    axs[1, idx].legend()

plt.xlabel("Min samples")
plt.show()'''

dbscan = cluster.DBSCAN(eps=2.5, min_samples=10).fit(df_centered)
print(f"Количество кластеров={num_of_labels(dbscan)}")
plot(dbscan.labels_)

dbscan = cluster.DBSCAN(eps=0.5, min_samples=2).fit(df_centered)
print(f"Количество кластеров={num_of_labels(dbscan)}")
plot(dbscan.labels_)

dbscan = cluster.DBSCAN(eps=1.5, min_samples=6).fit(df_centered)
print(f"Количество кластеров={num_of_labels(dbscan)}")
plot(dbscan.labels_)
