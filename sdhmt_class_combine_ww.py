import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches

# 1 get SDHMT data
# read SDHMT data
sdhmt = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\2025_sdhmt_2Q.csv", low_memory = False)
mm_product = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\mm_code_product.csv", low_memory = False)
sdhmt_df = pd.merge(sdhmt, mm_product, on="PRODUCT@SORT@WAFER")

# Group by the specified columns and calculate the sums
# Define a function to sum multiple columns
def sum_columns(group, columns):
    return group[columns].sum().sum()

# Group by the specified columns
grouped = sdhmt_df.groupby(["SITE@NTSC@FAR@852061", "mm_code", "SDHMT_WW"])

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

# read class data
# recent 3 month class data
class_1 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\METEOR LAKE-BASE-P682_Class_Test.csv", low_memory = False)
class_2 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\METEOR LAKE-BASE-P281_Class_Test.csv", low_memory = False)
class_3 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\LUNAR LAKE-BASE-M442_Class_Test.csv", low_memory = False)
class_4 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-S816_Class_Test.csv", low_memory = False)
class_5 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-P281_Class_Test.csv", low_memory = False)
class_6 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-H682_Class_Test.csv", low_memory = False)
# Q1 class data
class_7 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\METEOR LAKE-BASE-P682_Class_Test.csv", low_memory = False)
class_8 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\METEOR LAKE-BASE-P281_Class_Test.csv", low_memory = False)
class_9 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\LUNAR LAKE-BASE-M442_Class_Test.csv", low_memory = False)
class_10 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-S816_Class_Test.csv", low_memory = False)
class_11 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-P281_Class_Test.csv", low_memory = False)
class_12 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-H682_Class_Test.csv", low_memory = False)

# 2 select the useful columns, rename, and concat all products together
class_raw = pd.concat([class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10, class_11, class_12], ignore_index=True)
class_raw = class_raw.drop_duplicates()

# Create the new column 'class_ibin' using fillna, get data from 7820, if it is NA, then get data from 6248
class_raw['class_ibin'] = class_raw['ul#test#7820_classhot#last#interface_bin'].fillna(
    class_raw['ul#test#6248_classhot#last#interface_bin']
)
class_raw['class_fbin'] = class_raw['ul#test#7820_classhot#last#functional_bin'].fillna(
    class_raw['ul#test#6248_classhot#last#functional_bin']
)

# check ww range
# Get the minimum and maximum values of the 'test_end_ww' column
min_value = class_raw['test_end_ww'].min()
max_value = class_raw['test_end_ww'].max()

# Display the range
print(f"The range of 'test_end_ww' is from {min_value} to {max_value}.")

# useful columns
useful_columns = ['class_ibin', 'class_fbin', 'ul#test#197254#last#functional_bin', 'ul#test#197254#last#interface_bin', 'sub_unit_fab_lot', 'sub_unit_wafer_id', 'sub_unit_x_location', 'sub_unit_y_location', 'll#opinfo#852061#facility', 'mm_code', 'test_end_ww']
df = class_raw[useful_columns]

# Create a new column 'group_SDHMT' based on the value of 'ul#test#197254#last#functional_bin'
df['group_SDHMT'] = np.where(df['ul#test#197254#last#functional_bin'] == 101, '101',
                             np.where(df['ul#test#197254#last#functional_bin'] == 693, 'skip', 'skip'))
df = df.rename(columns={
    'll#opinfo#852061#facility': 'WLA_site',
})

df['class_fbin'] = df['class_fbin'].astype(str)
df['class_ibin'] = df['class_ibin'].astype(str)
print(df.columns)
print(df['group_SDHMT'].unique())

# group by ibin
grouped_df = df.groupby(['WLA_site', 'mm_code', 'group_SDHMT', 'test_end_ww', 'class_ibin']).size().reset_index(name='count')
# drop rows with NA in column 'class_ibin'

grouped_df = grouped_df[grouped_df['class_ibin'] != 'nan']

# Pivot the DataFrame
split_df = grouped_df.pivot_table(
    index=['WLA_site', 'mm_code', 'group_SDHMT', 'test_end_ww'],
    columns='class_ibin',
    values='count',
    fill_value=0
).reset_index()
print(split_df.dtypes)

# Calculate the total tested number at class. do we need to exclude NA? not tested?
split_df['total_tested'] = split_df.iloc[:, 4:].sum(axis=1)

# group by fbin
grouped_df_fbin = df.groupby(['WLA_site', 'mm_code', 'group_SDHMT', 'test_end_ww', 'class_fbin']).size().reset_index(name='count')
grouped_df_fbin = grouped_df_fbin[grouped_df_fbin['class_fbin'] != 'nan']
# Pivot the DataFrame
split_df_fbin = grouped_df_fbin.pivot_table(
    index=['WLA_site', 'mm_code', 'group_SDHMT', 'test_end_ww'],
    columns='class_fbin',
    values='count',
    fill_value=0
).reset_index()

# merge class ibin and fbin
class_merge = pd.merge(
    split_df,
    split_df_fbin,
    on=['WLA_site', 'mm_code', 'group_SDHMT', 'test_end_ww'],
    how='inner'
)

# First, ensure 'group_SDHMT' is treated as a categorical variable
class_merge['group_SDHMT'] = class_merge['group_SDHMT'].astype(str)

# Pivot the DataFrame
pivoted_df = class_merge.pivot_table(
    index=['WLA_site', 'mm_code', 'test_end_ww'],
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
print(all_bin)

# calculate overal class bin% based on SDHMT skip and 101 conditions.
# Iterate over each bin and perform the calculation
for bin_value in all_bin:
    # Create the new column name
    new_column_name = f'class_{bin_value}'
    # Perform the calculation
    pivoted_df[new_column_name] = (
            pivoted_df[f'{bin_value}_skip'] / pivoted_df['total_tested_skip'] -
            pivoted_df[f'{bin_value}_101'] / pivoted_df['total_tested_101']
    )

# deal with FB9XX based on different products
bin9XX = ['902.0', '912.0', '900.0']

# Iterate over each bin and perform the calculation
for bin_value in bin9XX:
    # Create the new column name
    new_column_name = f'class_{bin_value}'

    # Perform the calculation
    pivoted_df[new_column_name] = (
            (pivoted_df[f'{bin_value}_101'] + pivoted_df[f'{bin_value}_skip']) /
            pivoted_df['class_total_tested']
    )

product_map = {
'METEOR LAKE-BASE-P281':'902.0',
'METEOR LAKE-BASE-P682':'902.0',
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

# Apply the function to each row
pivoted_df['class_9.0_excluded'] = pivoted_df.apply(calculate_class_9_excluded, axis=1)

# calculate class_WIYL based products.
'''
MTL682,	WIYL =Skip die (PIYL+B36+B9~902) % - SDHMT Bin1 (PIYL+B36+B9~902)% + (Total B902/Total Tested)
MTL281,	WIYL =Skip die (PIYL+B36+B9~902) % - SDHMT Bin1 (PIYL+B36+B9~902)% + (Total B902/Total Tested)
ARL816L,	WIYL =Skip die (PIYL+B36+B9~912) % - SDHMT Bin1 (PIYL+B36+B9~912)% + (Total B912/Total Tested)
ARL816B,	WIYL =Skip die (PIYL+B36+B9~912) % - SDHMT Bin1 (PIYL+B36+B9~912)% + (Total B912/Total Tested)
ARL682,	WIYL =Skip die (PIYL+B36+B9~902) % - SDHMT Bin1 (PIYL+B36+B9~902)% + (Total B902/Total Tested)
ARL281,	WIYL =Skip die (PIYL+B36+B9~902) % - SDHMT Bin1 (PIYL+B36+B9~902)% + (Total B902/Total Tested)
ARL681,	WIYL =Skip die (PIYL+B36+B9~912) % - SDHMT Bin1 (PIYL+B36+B9~912)% + (Total B912/Total Tested)
LNL,	WIYL =Skip die (PIYL+B36+B9~900) % - SDHMT Bin1 (PIYL+B36+B9~900)% + (Total B900/Total Tested)
'''
piyl_bin_added = ['8.0', '10.0', '15.0', '53.0', '97.0', '98.0', '99.0', '36.0', '9.0']

# Function to calculate 'class_WIYL'
def calculate_class_WIYL(row):
    mm_code = row['mm_code']
    if mm_code in product_map:
        bin_value = product_map[mm_code]

        # Sum the specified columns for _skip and _101
        sum_skip = sum(row[f'{bin}_skip'] for bin in piyl_bin_added)
        sum_101 = sum(row[f'{bin}_101'] for bin in piyl_bin_added)

        # Perform the calculation
        result = (
                (sum_skip - row[f'{bin_value}_skip']) / row['total_tested_skip'] -
                (sum_101 - row[f'{bin_value}_101']) / row['total_tested_101'] +
                row[f'class_{bin_value}']
        )
        return result

    return np.nan  # Default value if no condition is met

# Apply the function to each row
pivoted_df['class_WIYL'] = pivoted_df.apply(calculate_class_WIYL, axis=1)
pivoted_df['class_sdhmt_sampling'] = pivoted_df['total_tested_101']/pivoted_df['class_total_tested']
pivoted_df['class_SBB'] = pivoted_df['class_8.0'] + pivoted_df['class_10.0'] + pivoted_df['class_36.0'] + pivoted_df['class_99.0']
# Function to calculate 'class_Si_remaining'
def calculate_class_Si_remaining(row):
    mm_code = row['mm_code']
    if mm_code in product_map:
        # Use the mapped bin value from product_map
        bin_value = product_map[mm_code]
        result = row['class_97.0'] + row[f'class_{bin_value}']
        return result

    return np.nan  # Default value if no condition is met

# Apply the function to each row
pivoted_df['class_Si_remaining'] = pivoted_df.apply(calculate_class_Si_remaining, axis=1)

# get the useful columns for class data
class_columns = [col for col in pivoted_df.columns if col.startswith('class')]
class_conlumns_filter = ['WLA_site', 'mm_code', 'test_end_ww'] + class_columns
print(class_conlumns_filter)

# 3, merge SDHMT and class by 'site', 'mm_code', 'ww'
# get the class data with useful columns
class_short_df = pivoted_df[class_conlumns_filter]
# change both tables ww to string
sdhmt_cal['SDHMT_WW'] = sdhmt_cal['SDHMT_WW'].astype(str)
class_short_df['test_end_ww'] = class_short_df['test_end_ww'].astype(str)
# change both table the site info the same format
class_short_df['WLA_site'] = class_short_df['WLA_site'].replace({'F021': 'F21', 'RA3': 'D1V'})
print(sdhmt_cal['SITE@NTSC@FAR@852061'].unique())
print(class_short_df['WLA_site'].unique())
print(sdhmt_cal['mm_code'].unique())
print(class_short_df['mm_code'].unique())
print(sdhmt_cal['SDHMT_WW'].unique())
print(class_short_df['test_end_ww'].unique())
class_short_df['test_end_ww'] = class_short_df['test_end_ww'].str.replace('.0', '', regex=False)

# merge 2 tables
combine_sdhmt_class = pd.merge(sdhmt_cal, class_short_df, left_on = ['SITE@NTSC@FAR@852061', 'mm_code', 'SDHMT_WW'], right_on = ['WLA_site', 'mm_code', 'test_end_ww'], how = 'inner')
# Rename the columns
combine_sdhmt_class = combine_sdhmt_class.rename(columns={
    'SITE@NTSC@FAR@852061': 'site',
    'SDHMT_WW': 'WW'
})

# Filter rows where 'WW' starts with '2025'
combine_sdhmt_class = combine_sdhmt_class[combine_sdhmt_class['WW'].str.startswith('2025')]
print(combine_sdhmt_class.columns)
combine_sdhmt_class['total_WLA_EYL'] = combine_sdhmt_class['yield_loss_sdhmt%']*combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_WIYL']*(1-combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_WLA_EY'] = 1 - combine_sdhmt_class['total_WLA_EYL']
combine_sdhmt_class['total_SBB'] = combine_sdhmt_class['SBB_sdhmt%']*combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_SBB']*(1-combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_Si_remainig'] = combine_sdhmt_class['Si_remaining_sdhmt%']*combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_Si_remaining']*(1-combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_ULT'] = combine_sdhmt_class['ULT_sdhmt%']*combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_53.0']*(1-combine_sdhmt_class['class_sdhmt_sampling'])
combine_sdhmt_class['total_IOE_chipping'] = combine_sdhmt_class['IOE_chipping_sdhmt%']*combine_sdhmt_class['class_sdhmt_sampling'] + combine_sdhmt_class['class_1594.0']*(1-combine_sdhmt_class['class_sdhmt_sampling'])
# replace any value <0 to 0, ['total_IOE_chipping'] , ['total_ULT'] , ['total_Si_remainig'],['total_SBB']
# List of columns to check for negative values
columns_to_check = ['total_IOE_chipping', 'total_ULT', 'total_Si_remainig', 'total_SBB']
# Replace negative values with 0
combine_sdhmt_class[columns_to_check] = combine_sdhmt_class[columns_to_check].clip(lower=0)
# save file
combine_sdhmt_class.to_csv('combine_sdhmt_class.csv', index=False)