import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the merged data
data = pd.read_csv('/home/ubuntu/process_whilst_in_greece_align_traindata.csv')

# Drop the 'type' column and rename the 'competitor' column to 'competitorID'
data = data.drop(columns=['type'])
data = data.rename(columns={'competitor': 'competitorID'})

# Organize by 'balance' column in descending order
data = data.sort_values(by='balance', ascending=False)

# Take the top 20% of the dataset
top_20_percent = int(0.2 * len(data))
data = data.head(top_20_percent)

# Drop NaNs
data = data.dropna()

# Map the 'decision' column
data['decision'] = data['decision'].map({'backer': 1, 'layer': 0})

# Organize by the 'time' column in ascending order
data = data.sort_values(by='time')
# Reorder columns to match the order the scaler was fitted on, including 'alignment'
column_order = ['competitorID', 'time', 'exchange', 'odds', 'agentID', 'stake', 'distance', 'rank', 'balance', 'decision','alignment']
data = data[column_order]



# Create a new scaler for the combined data
new_scaler = MinMaxScaler()
combined_data_scaled = new_scaler.fit_transform(data)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(combined_data_scaled, columns=list(data.columns))

# Save the new dataset
scaled_df.to_csv('/home/ubuntu/processed_align_train_new_greece.csv', index=False)

# Save the new scaler
joblib.dump(new_scaler, '/home/ubuntu/LSTM_BBE/align_minmaxscaler_newgreece.pkl')

print('Preprocessing complete. New dataset and scaler saved.')
