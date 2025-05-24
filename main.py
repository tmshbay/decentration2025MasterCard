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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Recency — время до последней транзакции
last_trans = df.groupby('card_id')['transaction_timestamp'].max().reset_index()
today = datetime.now()
last_trans['recency_days'] = (today - last_trans['transaction_timestamp']).dt.days
rfm = last_trans[['card_id', 'recency_days']]

# Frequency — общее количество транзакций
freq = df.groupby('card_id').size().reset_index(name='frequency')

# Monetary — средняя сумма транзакции
monetary = df.groupby('card_id')['transaction_amount_kzt'].mean().reset_index(name='avg_transaction_amount')

# Частота транзакций (среднее число дней между операциями)
df_sorted = df.sort_values(by=['card_id', 'transaction_timestamp'])
df_sorted['next_trans'] = df_sorted.groupby('card_id')['transaction_timestamp'].shift(-1)
df_sorted['days_between'] = (df_sorted['next_trans'] - df_sorted['transaction_timestamp']).dt.days
avg_gap = df_sorted.groupby('card_id')['days_between'].mean().reset_index(name='avg_days_between_txns')

# Доля безналичных транзакций
contactless = df.groupby('card_id')['pos_entry_mode'].apply(lambda x: (x == 'Contactless').sum() / len(x)).reset_index(name='pct_contactless')

# Доля использования кошельков
digital_wallets = df.groupby('card_id')['wallet_type'].apply(lambda x: x.notnull().sum() / len(x)).reset_index(name='pct_digital_wallet')

# Географическая активность — кол-во городов
geo_activity = df.groupby('card_id')['merchant_city'].nunique().reset_index(name='num_cities_visited')

# Самый популярный город
top_city = df.groupby('card_id')['merchant_city'].agg(lambda x: x.value_counts().index[0] if not x.isna().all() else 'Unknown').reset_index(name='top_city')

# Предпочтения по MCC — самая популярная категория
top_mcc = df.groupby('card_id')['merchant_mcc'].agg(lambda x: x.value_counts().index[0] if not x.isna().all() else 'Unknown').reset_index(name='top_mcc')

# Распределение трат по MCC
mcc_distribution = pd.pivot_table(df, index='card_id', columns='merchant_mcc', values='transaction_amount_kzt', aggfunc='count', fill_value=0)
mcc_distribution = mcc_distribution.div(mcc_distribution.sum(axis=1), axis=0).add_prefix('mcc_')

# Объединение всех фичей
features = rfm \
    .merge(freq, on='card_id') \
    .merge(monetary, on='card_id') \
    .merge(avg_gap, on='card_id') \
    .merge(contactless, on='card_id') \
    .merge(digital_wallets, on='card_id') \
    .merge(geo_activity, on='card_id') \
    .merge(top_city, on='card_id') \
    .merge(top_mcc, on='card_id') \
    .merge(mcc_distribution, on='card_id')

# Удаление строк с NaN
features = features.dropna()

# Приводим top_mcc к строкам, чтобы избежать ошибки
features['top_mcc'] = features['top_mcc'].astype(str)

# Сохраняем card_id для дальнейшего сопоставления
X = features.drop(columns=['card_id', 'top_city', 'top_mcc'])

# One-Hot Encoding для top_city и top_mcc
city_encoded = pd.get_dummies(features['top_city'], prefix='city')
mcc_encoded = pd.get_dummies(features['top_mcc'], prefix='mcc_top')

# Объединение с числовыми фичами
X_final = pd.concat([X, city_encoded, mcc_encoded], axis=1)

# Нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# %%
# Кластеризация
print("Обучение модели KMeans...")
kmeans = KMeans(n_clusters=20, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# %%
# Оценка силуэта
silhouette_km = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Score KMeans: {silhouette_km:.3f}")

# %%
# Автоматическая интерпретация сегментов

tx_high = features['frequency'].quantile(0.66)
tx_low = features['frequency'].quantile(0.33)

amt_high = features['avg_transaction_amount'].quantile(0.66)
amt_low = features['avg_transaction_amount'].quantile(0.33)

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

    if row['avg_transaction_amount'] > amt_high:
        name += "High-Spending"
    elif row['avg_transaction_amount'] < amt_low:
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
features['segment'] = kmeans_labels

segment_summary = features.groupby('segment').agg({
    'recency_days': ['mean', 'median'],
    'frequency': ['mean', 'median'],
    'avg_transaction_amount': ['mean', 'median'],
    'num_cities_visited': ['mean', 'median'],
    'card_id': 'count',
    'top_city': lambda x: ', '.join(set(x.astype(str))),
    'top_mcc': lambda x: ', '.join(set(x.astype(str)))
}).reset_index()

segment_summary.columns = [
    'segment',
    'recency_mean', 'recency_median',
    'frequency_mean', 'frequency_median',
    'monetary_mean', 'monetary_median',
    'cities_mean', 'cities_median',
    'clients',
    'top_cities',
    'top_mccs'
]

segment_summary['share'] = segment_summary['clients'] / segment_summary['clients'].sum()

# %%
# Добавляем дополнительные данные о клиентах
print("Добавление поведенческой информации...")

# Чаще всего используемый способ оплаты
features['most_used_payment_method'] = df.groupby('card_id')['transaction_type'].agg(lambda x: x.value_counts().index[0]).reset_index()['transaction_type']

# Использовал ли цифровой кошелёк
features['digital_wallet_used'] = df.groupby('card_id')['wallet_type'].apply(lambda x: 'Yes' if x.notnull().any() else 'No').reset_index()['wallet_type']

# %%
# Сохранение результатов
print("Сохранение результатов...")

# Финальный DataFrame с полной информацией
final_output = features[[
    'card_id',
    'segment',
    'segment_description',
    'top_city',
    'top_mcc',
    'most_used_payment_method',
    'digital_wallet_used'
]]

# Сохраняем в CSV и Parquet
final_output.to_parquet('customer_segments.parquet', index=False)
final_output.to_csv('customer_segments.csv', index=False)

# Также сохраняем сводку по сегментам
segment_summary.to_csv('segment_interpretation.csv', index=False)

# Словарь признаков — автоматически
descriptions = []
sources = []

for col in X_final.columns:
    if col.startswith('city_'):
        descriptions.append(f'Preferred city: {col.replace("city_", "", 1)}')
        sources.append('Geography')
    elif col.startswith('mcc_top_'):
        descriptions.append(f'Preferred MCC category: {col.replace("mcc_top_", "", 1)}')
        sources.append('Category')
    elif col == 'recency_days':
        descriptions.append('Days since last transaction (Recency)')
        sources.append('RFM')
    elif col == 'frequency':
        descriptions.append('Total number of transactions (Frequency)')
        sources.append('RFM')
    elif col == 'avg_transaction_amount':
        descriptions.append('Average transaction amount (Monetary)')
        sources.append('RFM')
    elif col == 'avg_days_between_txns':
        descriptions.append('Average days between transactions')
        sources.append('Original feature')
    elif col == 'pct_contactless':
        descriptions.append('Percentage of contactless payments')
        sources.append('Original feature')
    elif col == 'pct_digital_wallet':
        descriptions.append('Percentage of digital wallet usage')
        sources.append('Original feature')
    elif col == 'num_cities_visited':
        descriptions.append('Number of different cities visited')
        sources.append('Original feature')
    elif col.startswith('mcc_'):
        descriptions.append(f'Transaction count in MCC category {col.replace("mcc_", "", 1)}')
        sources.append('MCC distribution')
    else:
        descriptions.append('Custom or unknown feature')
        sources.append('Unknown')

data_dict = pd.DataFrame({
    'feature': X_final.columns,
    'description': descriptions,
    'source': sources
})

data_dict.to_csv('data_dictionary_features.csv', index=False)

print("✅ Все файлы успешно сохранены!")