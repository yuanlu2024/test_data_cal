import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import pandas as pd
import numpy as np

# 1. Get SDHMT data
# Read SDHMT data
sdhmt = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\2025_sdhmt_2Q.csv", low_memory=False)
mm_product = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\mm_code_product.csv", low_memory=False)
sdhmt_df = pd.merge(sdhmt, mm_product, on="PRODUCT@SORT@WAFER")

# Group by the specified columns and calculate the sums
# Define a function to sum multiple columns
def sum_columns(group, columns):
    return group[columns].sum().sum()

# Group by the specified columns, excluding 'SDHMT_WW'
grouped = sdhmt_df.groupby(["SITE@NTSC@FAR@852061", "mm_code"])

# Calculate the sums using apply
sdhmt_cal = grouped.apply(lambda x: pd.Series({
    'IB1': x['IB1@SDT@WAFER'].sum(),
    'FB693': x['FB693@SDT@WAFER'].sum(),
    'FB9301': x['FB9301@SDT@WAFER'].sum(),
    'FB1591': x['FB1591@SDT@WAFER'].sum(),
    'ULT': sum_columns(x, ["IB31@SDT@WAFER", "IB98@SDT@WAFER"]),
    'Si_remaining': sum_columns(x, ["IB9@SDT@WAFER", "IB97@SDT@WAFER"]),
    'SBB': sum_columns(x, ["IB8@SDT@WAFER", "IB10@SDT@WAFER", "IB36@SDT@WAFER", "IB99@SDT@WAFER"]),
    'N_TESTED': x['N_TESTED@SORT@WAFER'].sum()
})).reset_index()

# Calculate 'total_tested' as the denominator for the new columns
sdhmt_cal['total_tested'] = sdhmt_cal['N_TESTED'] - sdhmt_cal['FB693'] - sdhmt_cal['FB9301']

# Add new columns to the DataFrame using 'total_tested' as the denominator
sdhmt_cal['yield_sdhmt%'] = sdhmt_cal['IB1'] / sdhmt_cal['total_tested']
sdhmt_cal['SBB_sdhmt%'] = sdhmt_cal['SBB'] / sdhmt_cal['total_tested']
sdhmt_cal['ULT_sdhmt%'] = sdhmt_cal['ULT'] / sdhmt_cal['total_tested']
sdhmt_cal['IOE_chipping_sdhmt%'] = sdhmt_cal['FB1591'] / sdhmt_cal['total_tested']
sdhmt_cal['Si_remaining_sdhmt%'] = sdhmt_cal['Si_remaining'] / sdhmt_cal['total_tested']
sdhmt_cal['yield_loss_sdhmt%'] = 1 - sdhmt_cal['yield_sdhmt%']

# Read class data
# Recent 3 month class data
class_files = [
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\METEOR LAKE-BASE-P682_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\METEOR LAKE-BASE-P281_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\LUNAR LAKE-BASE-M442_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-S816_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-P281_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-H682_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\METEOR LAKE-BASE-P682_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\METEOR LAKE-BASE-P281_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\LUNAR LAKE-BASE-M442_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-S816_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-P281_Class_Test.csv",
    r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-H682_Class_Test.csv"
]

class_raw = pd.concat([pd.read_csv(file, low_memory=False) for file in class_files], ignore_index=True)
class_raw = class_raw.drop_duplicates()

# Create the new column 'class_ibin' using fillna
class_raw['class_ibin'] = class_raw['ul#test#7820_classhot#last#interface_bin'].fillna(
    class_raw['ul#test#6248_classhot#last#interface_bin']
)
class_raw['class_fbin'] = class_raw['ul#test#7820_classhot#last#functional_bin'].fillna(
    class_raw['ul#test#6248_classhot#last#functional_bin']
)

# Useful columns
useful_columns = ['class_ibin', 'class_fbin', 'ul#test#197254#last#functional_bin', 'ul#test#197254#last#interface_bin', 'sub_unit_fab_lot', 'sub_unit_wafer_id', 'sub_unit_x_location', 'sub_unit_y_location', 'll#opinfo#852061#facility', 'mm_code']
df = class_raw[useful_columns]

# Create a new column 'group_SDHMT' based on the value of 'ul#test#197254#last#functional_bin'
df['group_SDHMT'] = np.where(df['ul#test#197254#last#functional_bin'] == 101, '101',
                             np.where(df['ul#test#197254#last#functional_bin'] == 693, 'skip', 'skip'))
df = df.rename(columns={
    'll#opinfo#852061#facility': 'WLA_site',
})

df['class_fbin'] = df['class_fbin'].astype(str)
df['class_ibin'] = df['class_ibin'].astype(str)

# Group by ibin, excluding 'test_end_ww'
grouped_df = df.groupby(['WLA_site', 'mm_code', 'group_SDHMT', 'class_ibin']).size().reset_index(name='count')
grouped_df = grouped_df[grouped_df['class_ibin'] != 'nan']

# Pivot the DataFrame
split_df = grouped_df.pivot_table(
    index=['WLA_site', 'mm_code', 'group_SDHMT'],
    columns='class_ibin',
    values='count',
    fill_value=0
).reset_index()

# Calculate the total tested number at class
split_df['total_tested'] = split_df.iloc[:, 3:].sum(axis=1)

# Group by fbin, excluding 'test_end_ww'
grouped_df_fbin = df.groupby(['WLA_site', 'mm_code', 'group_SDHMT', 'class_fbin']).size().reset_index(name='count')
grouped_df_fbin = grouped_df_fbin[grouped_df_fbin['class_fbin'] != 'nan']

# Pivot the DataFrame
split_df_fbin = grouped_df_fbin.pivot_table(
    index=['WLA_site', 'mm_code', 'group_SDHMT'],
    columns='class_fbin',
    values='count',
    fill_value=0
).reset_index()

# Merge class ibin and fbin
class_merge = pd.merge(
    split_df,
    split_df_fbin,
    on=['WLA_site', 'mm_code', 'group_SDHMT'],
    how='inner'
)

# First, ensure 'group_SDHMT' is treated as a categorical variable
class_merge['group_SDHMT'] = class_merge['group_SDHMT'].astype(str)

# Pivot the DataFrame
pivoted_df = class_merge.pivot_table(
    index=['WLA_site', 'mm_code'],
    columns='group_SDHMT',
    aggfunc='sum'  # or another aggregation function if needed
)

# Flatten the MultiIndex columns
pivoted_df.columns = ['_'.join(map(str, col)).strip() for col in pivoted_df.columns.values]

# Reset index to make it a regular DataFrame
pivoted_df = pivoted_df.reset_index()
pivoted_df['class_total_tested'] = pivoted_df['total_tested_101'] + pivoted_df['total_tested_skip']

# PIYL BIN:
piyl_bin = ['8.0', '10.0', '15.0', '53.0', '97.0', '98.0', '99.0']
all_bin = piyl_bin + ['36.0', '1594.0']

# Calculate overall class bin% based on SDHMT skip and 101 conditions.
for bin_value in all_bin:
    new_column_name = f'class_{bin_value}'
    pivoted_df[new_column_name] = (
        pivoted_df[f'{bin_value}_skip'] / pivoted_df['total_tested_skip'] -
        pivoted_df[f'{bin_value}_101'] / pivoted_df['total_tested_101']
    )

# Deal with FB9XX based on different products
bin9XX = ['902.0', '912.0', '900.0']

for bin_value in bin9XX:
    new_column_name = f'class_{bin_value}'
    pivoted_df[new_column_name] = (
        (pivoted_df[f'{bin_value}_101'] + pivoted_df[f'{bin_value}_skip']) /
        pivoted_df['class_total_tested']
    )

product_map = {
    'METEOR LAKE-BASE-P281': '902.0',
    'METEOR LAKE-BASE-P682': '902.0',
    'ARROW LAKE-BASE-H682': '902.0',
    'ARROW LAKE-BASE-P281': '902.0',
    'ARROW LAKE-BASE-S816': '912.0',
    'LUNAR LAKE-BASE-M442': '900.0',
    'ARROW LAKE-BASE-H681': '912.0'
}

# Function to calculate 'class_9.0_excluded'
def calculate_class_9_excluded(row):
    mm_code = row['mm_code']
    if mm_code in product_map:
        bin_value = product_map[mm_code]
        return (
            (row['9.0_skip'] - row[f'{bin_value}_skip']) / row['total_tested_skip'] -
            (row['9.0_101'] - row[f'{bin_value}_101']) / row['total_tested_101']
        )
    return np.nan  # Default value if no condition is met

pivoted_df['class_9.0_excluded'] = pivoted_df.apply(calculate_class_9_excluded, axis=1)

# Calculate class_WIYL based on products
piyl_bin_added = ['8.0', '10.0', '15.0', '53.0', '97.0', '98.0', '99.0', '36.0', '9.0']

def calculate_class_WIYL(row):
    mm_code = row['mm_code']
    if mm_code in product_map:
        bin_value = product_map[mm_code]
        sum_skip = sum(row[f'{bin}_skip'] for bin in piyl_bin_added)
        sum_101 = sum(row[f'{bin}_101'] for bin in piyl_bin_added)
        result = (
            (sum_skip - row[f'{bin_value}_skip']) / row['total_tested_skip'] -
            (sum_101 - row[f'{bin_value}_101']) / row['total_tested_101'] +
            row[f'class_{bin_value}']
        )
        return result
    return np.nan  # Default value if no condition is met

pivoted_df['class_WIYL'] = pivoted_df.apply(calculate_class_WIYL, axis=1)
pivoted_df['class_sdhmt_sampling'] = pivoted_df['total_tested_101'] / pivoted_df['class_total_tested']
pivoted_df['class_SBB'] = pivoted_df['class_8.0'] + pivoted_df['class_10.0'] + pivoted_df['class_36.0'] + pivoted_df['class_99.0']

def calculate_class_Si_remaining(row):
    mm_code = row['mm_code']
    if mm_code in product_map:
        bin_value = product_map[mm_code]
        result = row['class_97.0'] + row[f'class_{bin_value}']
        return result
    return np.nan  # Default value if no condition is met

pivoted_df['class_Si_remaining'] = pivoted_df.apply(calculate_class_Si_remaining, axis=1)

# Get the useful columns for class data
class_columns = [col for col in pivoted_df.columns if col.startswith('class')]
class_columns_filter = ['WLA_site', 'mm_code'] + class_columns

# Merge SDHMT and class by 'site' and 'mm_code'
class_short_df = pivoted_df[class_columns_filter]
sdhmt_cal['SITE@NTSC@FAR@852061'] = sdhmt_cal['SITE@NTSC@FAR@852061'].astype(str)
class_short_df['WLA_site'] = class_short_df['WLA_site'].replace({'F021': 'F21', 'RA3': 'D1V'})

# Merge 2 tables
combine_sdhmt_class = pd.merge(sdhmt_cal, class_short_df, left_on=['SITE@NTSC@FAR@852061', 'mm_code'], right_on=['WLA_site', 'mm_code'], how='inner')

# Rename the columns
combine_sdhmt_class = combine_sdhmt_class.rename(columns={
    'SITE@NTSC@FAR@852061': 'site'
})

# Calculate final metrics
combine_sdhmt_class['total_WLA_EYL'] = combine_sdhmt_class['yield_loss_sdhmt%'] * combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_WIYL'] * (1 - combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_WLA_EY'] = 1 - combine_sdhmt_class['total_WLA_EYL']
combine_sdhmt_class['total_SBB'] = combine_sdhmt_class['SBB_sdhmt%'] * combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_SBB'] * (1 - combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_Si_remaining'] = combine_sdhmt_class['Si_remaining_sdhmt%'] * combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_Si_remaining'] * (1 - combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_ULT'] = combine_sdhmt_class['ULT_sdhmt%'] * combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_53.0'] * (1 - combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_IOE_chipping'] = combine_sdhmt_class['IOE_chipping_sdhmt%'] * combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_1594.0'] * (1 - combine_sdhmt_class['class_sdhmt_sampling'])

# Save the final table to CSV
combine_sdhmt_class.to_csv('combine_sdhmt_class_final.csv', index=False)