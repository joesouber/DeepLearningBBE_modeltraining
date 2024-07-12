import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Function to extract top 20% data based on balance
def extract_top_20_percent(file_path, chunksize=100000):
    chunks = pd.read_csv(file_path, chunksize=chunksize)
    top_chunks = []

    for chunk in chunks:
        chunk_sorted = chunk.sort_values(by='balance', ascending=False)
        top_20_count = int(len(chunk_sorted) * 0.2)
        top_chunk = chunk_sorted.head(top_20_count)
        top_chunks.append(top_chunk)

    top_20_df = pd.concat(top_chunks)
    top_20_df = top_20_df.sort_values(by='balance', ascending=False).head(int(len(top_20_df) * 0.2))
    return top_20_df

# Function to extract bottom 20% data based on balance
def extract_bottom_20_percent(file_path, chunksize=100000):
    chunks = pd.read_csv(file_path, chunksize=chunksize)
    bottom_chunks = []

    for chunk in chunks:
        chunk_sorted = chunk.sort_values(by='balance', ascending=True)
        bottom_20_count = int(len(chunk_sorted) * 0.2)
        bottom_chunk = chunk_sorted.head(bottom_20_count)
        bottom_chunks.append(bottom_chunk)

    bottom_20_df = pd.concat(bottom_chunks)
    bottom_20_df = bottom_20_df.sort_values(by='balance', ascending=True).head(int(len(bottom_20_df) * 0.2))
    return bottom_20_df

# Function to preprocess data
def preprocess_data(df, features_to_drop):
    df.drop(columns=features_to_drop, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df['decision'] = df['decision'].map({'backer': 1, 'layer': 0})
    df.sort_values(by='time', inplace=True)
    return df

# Load and preprocess data
file_path = '/home/ubuntu/final_merged_combined_6.csv'
features_to_drop = ['type', 'competitorID', 'agentID', 'odds', 'exchange']  # Adjust this list to drop more features if needed

# Extract top and bottom 20% for pre-training
top_20_data = extract_top_20_percent(file_path)
bottom_20_data = extract_bottom_20_percent(file_path)
pretrain_data = pd.concat([top_20_data, bottom_20_data])
pretrain_data = preprocess_data(pretrain_data, features_to_drop)

# Extract only top 20% for final training
train_data = extract_top_20_percent(file_path)
train_data = preprocess_data(train_data, features_to_drop)

# Split the data into features and target
X_pretrain = pretrain_data.iloc[:, :-1].values
y_pretrain = pretrain_data.iloc[:, -1].values
X_train_final = train_data.iloc[:, :-1].values
y_train_final = train_data.iloc[:, -1].values

# Scale the pretrain data
scaler = MinMaxScaler()
X_pretrain = scaler.fit_transform(X_pretrain)

# Scale the final training data
X_train_final = scaler.transform(X_train_final)

# Save the scaler and preprocessed data
scaler_path = '/home/ubuntu/minmaxscaler_pretrain.joblib'
joblib.dump(scaler, scaler_path)
print(f'Scaler saved to {scaler_path}')

pretrain_data_path = '/home/ubuntu/pretrain_data.csv'
final_train_data_path = '/home/ubuntu/final_train_data.csv'

pd.DataFrame(X_pretrain).to_csv(pretrain_data_path, index=False)
pd.DataFrame(y_pretrain).to_csv('/home/ubuntu/pretrain_labels.csv', index=False)

pd.DataFrame(X_train_final).to_csv(final_train_data_path, index=False)
pd.DataFrame(y_train_final).to_csv('/home/ubuntu/final_train_labels.csv', index=False)

print(f'Pretrain data saved to {pretrain_data_path}')
print(f'Final train data saved to {final_train_data_path}')
