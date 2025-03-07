# %%
import pandas as pd
df = pd.read_csv("results/outputs/outputs.csv")

# %%
toy_df = df[df['Problem'] == 'Toy']
results = pd.pivot_table(toy_df, index='Metrics', columns='Method',
                         values='value', aggfunc=['mean', 'std']).reset_index().round(3)
print(results)
results.to_csv("tables/table1_part1.csv")

# %%
toy_df = df[df['Problem'] == 'Lorenz']
results = pd.pivot_table(toy_df, index='Metrics', columns='Method',
                         values='value', aggfunc=['mean', 'std']).reset_index().round(3)
print(results)
results.to_csv("tables/table1_part2.csv")

# %%
real_df = df[~df['Problem'].isin(['Toy', 'Lorenz'])]
real_df_short_term = real_df[real_df['Type'] == 'Short-term']
results = pd.pivot_table(real_df_short_term, index=['Problem'], columns=[
                         'Metrics', 'Method'], values='value', aggfunc=['mean']).reset_index().round(2)
print(results)
results.to_csv("tables/table3_part1.csv")

# %%
real_df = df[~df['Problem'].isin(['Toy', 'Lorenz'])]
real_df_long_term = real_df[real_df['Type'] == 'Long-term']
results = pd.pivot_table(real_df_long_term, index=['Problem'], columns=[
                         'Metrics', 'Method'], values='value', aggfunc=['mean']).reset_index().round(2)
print(results)
results.to_csv("tables/table3_part2.csv")

# %%


# %%
