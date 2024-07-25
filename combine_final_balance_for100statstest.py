import pandas as pd
import os

# Directory containing the CSV files
directory = "/content/XGBoost_TBBE/TBBE_OD_XGboost/Application"

# Agent configuration
agents = [
    ('Agent_Opinionated_Random', 10),
    ('Agent_Opinionated_Leader_Wins', 10),
    ('Agent_Opinionated_Underdog', 10),
    ('Agent_Opinionated_Back_Favourite', 10),
    ('Agent_Opinionated_Linex', 10),
    ('Agent_Opinionated_Priviledged', 5),
    ('XGBoostBettingAgent', 5),
    ('LSTMBettingAgent', 5)
]

# Reading all CSV files and processing each as a trial
all_trials_data = []

for i in range(101):
    file_path = os.path.join(directory, f"200_new_final_balance_{i}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Subtract 100,000,000 from each value
        df = df - 100_000_000

        # Calculating average sum for each betting agent
        agent_averages = {}
        start_index = 0

        for agent, count in agents:
            agent_data = df.iloc[:, start_index:start_index + count]
            agent_sum = agent_data.sum().sum()
            agent_average = agent_sum / (count * len(df))
            agent_averages[agent] = agent_average
            start_index += count

        all_trials_data.append(agent_averages)
    else:
        print(f"File not found: {file_path}")

# Creating a DataFrame for all trials
trials_df_1 = pd.DataFrame(all_trials_data)

# Display the res


#trials_df_1.to_csv("/content/final_balances_100_trials.csv", index = False)
trials_df_1
