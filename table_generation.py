import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='DS3M')
parser.add_argument('-train', '--train', action='store_true',
                    help='whether to retrain the model')

args = parser.parse_args()
restore = not args.train

if restore:
    df = pd.read_csv("results/outputs/outputs.csv")
else:
    df1 = pd.read_csv("results/outputs/outputs.csv")
    df1 = df1[df1['Method'] != '1 DS3M']
    df2 = pd.read_csv("results/outputs/outputs_generated.csv")
    df = pd.concat([df1, df2], ignore_index=True)

# %%
toy_df = df[df['Problem'] == 'Toy']
results = pd.pivot_table(toy_df, index='Metrics', columns='Method',
                         values='value', aggfunc=['mean', 'std'], dropna=True).reset_index().round(3)
print(results)
results.to_csv("tables/table1_part1.csv")

# %%
toy_df = df[df['Problem'] == 'Lorenz']
results = pd.pivot_table(toy_df, index='Metrics', columns='Method',
                         values='value', aggfunc=['mean', 'std'], dropna=True).reset_index().round(3)
print(results)
results.to_csv("tables/table1_part2.csv")

# %%
real_df = df[~df['Problem'].isin(['Toy', 'Lorenz'])]
real_df_short_term = real_df[real_df['Type'] == 'Short-term']
results = pd.pivot_table(real_df_short_term, index=['Problem'], columns=[
                         'Metrics', 'Method'], values='value', aggfunc=['mean'], dropna=True).reset_index().round(2)
print(results)
results.to_csv("tables/table3_part1.csv")

# %%
real_df = df[~df['Problem'].isin(['Toy', 'Lorenz'])]
real_df_long_term = real_df[real_df['Type'] == 'Long-term']
results = pd.pivot_table(real_df_long_term, index=['Problem'], columns=[
                         'Metrics', 'Method'], values='value', aggfunc=['mean'], dropna=True).reset_index().round(2)
print(results)
results.to_csv("tables/table3_part2.csv")

# %%
