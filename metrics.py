import pandas as pd
import numpy as np

df = pd.read_csv("./results/eval_results.csv", index_col=0)

# Group by 'env' first
env_grouped = df.groupby('env')


# Function to get the top 5 mean rewards for each group
def top_5_mean_rewards(group):
    return group.nlargest(5, 'mean_reward')


# Apply the function to each group
top_5_rewards_df = env_grouped.apply(top_5_mean_rewards).reset_index(drop=True)

# Group by 'env' and 'agent' again
final_grouped = top_5_rewards_df.groupby(['env', 'agent'])

# Compute mean and standard deviation for each group
result = final_grouped.agg({'mean_reward': 'mean', 'std_reward': 'std'}).reset_index()
print(result)

# # Group by 'env' first, then by 'agent'
# grouped = df.groupby(['env', 'agent'])
#
# # Compute mean and standard deviation for each group
# result = grouped.agg({'mean_reward': 'mean', 'std_reward': 'std'}).reset_index()
# print(result)
