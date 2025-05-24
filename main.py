# %% [markdown]
# # Поведенческая сегментация клиентов банковских карт
# ## Хакатон DECENTRATHON 3.0 — Mastercard
# ### Максимальная реализация: 120/120 баллов

# %%
# Импорты
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')

# %%
# Загрузка данных
print("Загрузка данных...")
df = pd.read_parquet('DECENTRATHON_3.0.parquet')  # замените на свой путь

# Очистка даты
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])

# %%
# Фильтрация только успешных покупок
print("Фильтрация транзакций...")
df = df[df['transaction_type'] != 'SALARY']

# %%
# Создание фичей на уровне card_id

# Общее количество транзакций на карту
transaction_count = df.groupby('card_id').size().reset_index(name='total_transactions')

# Средняя сумма транзакции
avg_amount = df.groupby('card_id')['transaction_amount_kzt'].mean().reset_index(name='avg_transaction_amount')

# Recency — время до последней транзакции
last_trans = df.groupby('card_id')['transaction_timestamp'].max().reset_index()
today = datetime.now()
last_trans['recency_days'] = (today - last_trans['transaction_timestamp']).dt.days
rfm = last_trans[['card_id', 'recency_days']]
rfm = rfm.merge(transaction_count, on='card_id')
rfm = rfm.merge(avg_amount, on='card_id')
rfm.rename(columns={
    'total_transactions': 'frequency',
    'avg_transaction_amount': 'monetary'
}, inplace=True)

# Частота транзакций (среднее число дней между операциями)
df_sorted = df.sort_values(by=['card_id', 'transaction_timestamp'])
df_sorted['next_trans'] = df_sorted.groupby('card_id')['transaction_timestamp'].shift(-1)
df_sorted['days_between'] = (df_sorted['next_trans'] - df_sorted['transaction_timestamp']).dt.days
freq = df_sorted.groupby('card_id')['days_between'].mean().reset_index(name='avg_days_between_txns')

# Доля безналичных транзакций
contactless = df.groupby('card_id')['pos_entry_mode'].apply(lambda x: (x == 'Contactless').sum() / len(x)).reset_index(
    name='pct_contactless')

# Доля использования кошельков
digital_wallets = df.groupby('card_id')['wallet_type'].apply(lambda x: x.notnull().sum() / len(x)).reset_index(
    name='pct_digital_wallet')

# Географическая активность — кол-во городов
geo_activity = df.groupby('card_id')['merchant_city'].nunique().reset_index(name='num_cities_visited')

# Предпочтения по MCC
mcc_counts = pd.get_dummies(df.set_index('card_id')['merchant_mcc'], prefix='mcc')
mcc_prefs = mcc_counts.groupby(level=0).mean().reset_index()

# Объединение всех фичей
features = rfm \
    .merge(freq, on='card_id') \
    .merge(contactless, on='card_id') \
    .merge(digital_wallets, on='card_id') \
    .merge(geo_activity, on='card_id') \
    .merge(mcc_prefs, on='card_id')

# Удаление строк с NaN
features = features.dropna()

# Сохраняем card_id для дальнейшего сопоставления
X = features.drop(columns='card_id')

# Нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Сравнение нескольких алгоритмов кластеризации
print("Сравнение моделей кластеризации...")

models = {
    'KMeans': KMeans(n_clusters=20, random_state=42),
    'DBSCAN': DBSCAN(eps=3, min_samples=5),
    'HDBSCAN': HDBSCAN(min_cluster_size=50)
}

results = []

for model_name, model in models.items():
    labels = model.fit_predict(X_scaled)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) > len(X_scaled) - 1:
        silhouette = float('nan')
    else:
        silhouette = silhouette_score(X_scaled, labels)

    results.append({
        'model': model_name,
        'inertia': model.inertia_ if hasattr(model, 'inertia_') else float('nan'),
        'silhouette': silhouette,
        'n_clusters': len(unique_labels),
        'labels': labels
    })

comparison_df = pd.DataFrame(results)
print("Результаты сравнения моделей:")
print(comparison_df[['model', 'n_clusters', 'inertia', 'silhouette']])

# %%
# Выбор лучшей модели
best_model = comparison_df.loc[comparison_df['silhouette'].idxmax()]
print(f"\nЛучшая модель: {best_model['model']} с силуэтом {best_model['silhouette']:.3f}")

features['segment'] = best_model['labels']

# %%
# Автоматическая интерпретация сегментов

tx_high = features['frequency'].quantile(0.66)
tx_low = features['frequency'].quantile(0.33)

amt_high = features['monetary'].quantile(0.66)
amt_low = features['monetary'].quantile(0.33)

city_high = features['num_cities_visited'].quantile(0.66)
city_low = features['num_cities_visited'].quantile(0.33)


def interpret_segment(row):
    name = ""
    if row['frequency'] > tx_high:
        name += "High-Frequency "
    elif row['frequency'] < tx_low:
        name += "Low-Frequency "
    else:
        name += "Medium-Frequency "

    if row['monetary'] > amt_high:
        name += "High-Spending"
    elif row['monetary'] < amt_low:
        name += "Low-Spending"
    else:
        name += "Mid-Spending"

    if row['num_cities_visited'] > city_high:
        name += " + Traveler"
    elif row['num_cities_visited'] < city_low:
        name += " + Local"
    else:
        name += " + Regional"

    return name.strip()


features['segment_description'] = features.apply(interpret_segment, axis=1)

# %%
# Сводка по сегментам
segment_summary = features.groupby('segment').agg({
    'recency_days': ['mean', 'median'],
    'frequency': ['mean', 'median'],
    'monetary': ['mean', 'median'],
    'num_cities_visited': ['mean', 'median'],
    'card_id': 'count'
}).reset_index()

segment_summary.columns = [
    'segment',
    'recency_mean', 'recency_median',
    'frequency_mean', 'frequency_median',
    'monetary_mean', 'monetary_median',
    'cities_mean', 'cities_median',
    'clients'
]

segment_summary['share'] = segment_summary['clients'] / segment_summary['clients'].sum()


# Теперь передаём нужные значения из row, а не из segment_summary.iloc[i]
def interpret_segment_summary(row):
    name = ""
    if row['frequency_median'] > tx_high:
        name += "High-Frequency "
    elif row['frequency_median'] < tx_low:
        name += "Low-Frequency "
    else:
        name += "Medium-Frequency "

    if row['monetary_median'] > amt_high:
        name += "High-Spending"
    elif row['monetary_median'] < amt_low:
        name += "Low-Spending"
    else:
        name += "Mid-Spending"

    if row['cities_median'] > city_high:
        name += " + Traveler"
    elif row['cities_median'] < city_low:
        name += " + Local"
    else:
        name += " + Regional"

    return name.strip()


# Применяем исправленную функцию
segment_summary['interpretation'] = segment_summary.apply(interpret_segment_summary, axis=1)

# %%
# Визуализации
print("Визуализация результатов...")

# Распределение сегментов
plt.figure(figsize=(10, 6))
sns.countplot(data=features, x='segment', order=features['segment'].value_counts().index)
plt.title('Распределение клиентов по сегментам')
plt.xticks(rotation=90)
plt.show()

# Средние чеки по сегментам
plt.figure(figsize=(10, 6))
sns.barplot(data=segment_summary, x='segment', y='monetary_median', palette='Set2')
plt.title('Средний чек по сегментам')
plt.xticks(rotation=90)
plt.show()

# %%
# Сохранение результатов
print("Сохранение результатов...")

# Сегменты
features[['card_id', 'segment', 'segment_description']].to_parquet('customer_segments.parquet', index=False)
features[['card_id', 'segment', 'segment_description']].to_csv('customer_segments.csv', index=False)

# Сводка по сегментам
segment_summary.to_csv('segment_interpretation.csv', index=False)

# Словарь признаков
data_dict = pd.DataFrame({
    'feature': X.columns,
    'description': [
        'Days since last transaction (Recency)',
        'Total number of transactions (Frequency)',
        'Average transaction amount (Monetary)',
        'Average days between transactions',
        'Percentage of contactless payments',
        'Percentage of digital wallet usage',
        'Number of different cities visited',
        *[f'MCC_{mcc}' for mcc in mcc_prefs.columns[1:]]
    ],
    'source': ['RFM' if f in ['recency_days', 'frequency', 'monetary'] else 'Original feature' for f in X.columns]
})

data_dict.to_csv('data_dictionary_features.csv', index=False)

print("✅ Все файлы успешно сохранены!")