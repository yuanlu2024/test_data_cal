import pandas as pd
# step1, import data
df1 = pd.read_csv("long_term_SOC_Trace_with_Test_4m.csv")
df2 = pd.read_csv("long_term_SOC_Trace_with_Test.csv")
df = pd.concat([df1, df2], ignore_index=True)

# step2, data clean, remove duplicate data
# drop rows with class data missing
df = df.dropna(subset=['ul#test#7820_classhot#last#interface_bin'])
# Check for duplicate rows
duplicates = df.duplicated()
# Count the number of duplicate rows
num_duplicates = duplicates.sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")
#drop duplicate rows:
df = df.drop_duplicates()

# only keep the last test results for each top die wafers:
# Convert 'lots_end_date' to datetime
df['lots_end_date'] = pd.to_datetime(df['lots_end_date'])
# check the time range of the dataset
date_min = df['lots_end_date'].min()
date_max = df['lots_end_date'].max()
# Find the maximum 'lots_end_date' for each group
max_dates = df.groupby(['top_die_lot', 'top_die_wafer_script'])['lots_end_date'].transform('max')
# Filter the DataFrame to keep only the rows with the latest test time for each group
df_latest = df[df['lots_end_date'] == max_dates]

# remove 'u1_u3', 'u1_u4'
values_to_drop = ['u1_u3', 'u1_u4']
# Filter the DataFrame to exclude rows with 'top_die_md_position' in the specified list
df_filter = df_latest[~df_latest['top_die_md_position'].isin(values_to_drop)]
df_filter.to_csv("0430_df_filter.csv", index = False)
'''
#check one top unite has # rows
top_die_index = df_filter.groupby(['top_die_lot', 'top_die_wafer_script', 'top_die_index']).size().reset_index(name='count')
#check the range
count_min = top_die_index['count'].min()
count_max = top_die_index['count'].max()
print(f"\nCount value range: {count_min} to {count_max}")
#check count = 2 data
top_die_index_2 = top_die_index[top_die_index['count']== 2]

top_die_2_df = pd.merge(
    df_filter,
    top_die_index_2,
    on=['top_die_lot', 'top_die_wafer_script', 'top_die_index'],
    how='inner'
)
top_die_2_df = top_die_2_df.sort_values(by='top_die_index')
'''

# calculate B15
grouped_df = df_filter.groupby(['top_die_lot', 'top_die_wafer_script', 'ul#test#7820_classhot#last#functional_bin']).size().reset_index(name='count')
grouped_df['ul#test#7820_classhot#last#functional_bin'] = grouped_df['ul#test#7820_classhot#last#functional_bin'].astype(str)
split_df = grouped_df.pivot_table(
    index=['top_die_lot', 'top_die_wafer_script'],
    columns='ul#test#7820_classhot#last#functional_bin',
    values='count',
    fill_value=0
).reset_index()
#print(split_df.dtypes)
#print(list(split_df.columns))
split_df['tested'] = split_df.iloc[:, 2:].sum(axis=1)
split_df['B15(FB1542+FB1562)'] = split_df['1542.0'] + split_df['1562.0']
split_df['B15%'] = split_df['B15(FB1542+FB1562)']/split_df['tested']*100

#find the class test time
df_filter['load_date_timestamp'] = pd.to_datetime(df_filter['load_date_timestamp'])
class_time = df_filter.groupby(['top_die_lot', 'top_die_wafer_script'])['load_date_timestamp'].max().reset_index()
class_df = pd.merge(split_df, class_time, on=['top_die_lot', 'top_die_wafer_script'], how='inner')
class_df.to_csv("class_df.csv", index = False)

'''
#find the outlier, some cal from WIJT
filters_0227 = pd.read_csv("filters_0227.csv")
outlier = pd.merge(df_filter, filters_0227, on=['top_die_lot', 'top_die_wafer_script'], how='inner')
outlier['ul#test#7820_classhot#last#functional_bin'] = outlier['ul#test#7820_classhot#last#functional_bin'].astype(str)
outlier_B15 = outlier[outlier['ul#test#7820_classhot#last#functional_bin'].str.contains('1542|1562')]
#outlier_B15.to_csv("outlier_B15.csv")

#check = df_filter.groupby(['ul#test#7820_classhot#last#functional_bin']).size().reset_index(name='count')
'''