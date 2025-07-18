import pandas as pd
import numpy as np

# read class data
# recent 3 month class data
df1 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\METEOR LAKE-BASE-P682_Class_Test.csv", low_memory = False)
df2 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\METEOR LAKE-BASE-P281_Class_Test.csv", low_memory = False)
df3 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\LUNAR LAKE-BASE-M442_Class_Test.csv", low_memory = False)
df4 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-S816_Class_Test.csv", low_memory = False)
df5 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-P281_Class_Test.csv", low_memory = False)
df6 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_4_7\ARROW LAKE-BASE-H682_Class_Test.csv", low_memory = False)
# Q1 class data
df7 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\METEOR LAKE-BASE-P682_Class_Test.csv", low_memory = False)
df8 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\METEOR LAKE-BASE-P281_Class_Test.csv", low_memory = False)
df9 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\LUNAR LAKE-BASE-M442_Class_Test.csv", low_memory = False)
df10 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-S816_Class_Test.csv", low_memory = False)
df11 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-P281_Class_Test.csv", low_memory = False)
df12 = pd.read_csv(r"\\atdsdwanalap1\E\MAOATM\Personal\yuanlu\weekly\0715\class_1_3\ARROW LAKE-BASE-H682_Class_Test.csv", low_memory = False)
# 1 lunar lake, class test step 6248
df3 = pd.concat([df3, df9], ignore_index=True)
df3 = df3.drop_duplicates()
# IB
lnl_grouped_ib = df3.groupby(['sub_unit_fab_lot', 'sub_unit_wafer_id', 'ul#test#6248_classhot#last#interface_bin']).size().reset_index(name='count')
lnl_grouped_ib['ul#test#6248_classhot#last#interface_bin'] = lnl_grouped_ib['ul#test#6248_classhot#last#interface_bin'].astype(str)
lnl_ib = lnl_grouped_ib.pivot_table(
    index=['sub_unit_fab_lot', 'sub_unit_wafer_id'],
    columns='ul#test#6248_classhot#last#interface_bin',
    values='count',
    fill_value=0
).reset_index()
# FB
lnl_grouped_fb = df3.groupby(['sub_unit_fab_lot', 'sub_unit_wafer_id', 'ul#test#6248_classhot#last#functional_bin']).size().reset_index(name='count')
lnl_grouped_fb['ul#test#6248_classhot#last#functional_bin'] = lnl_grouped_fb['ul#test#6248_classhot#last#functional_bin'].astype(str)
lnl_fb = lnl_grouped_fb.pivot_table(
    index=['sub_unit_fab_lot', 'sub_unit_wafer_id'],
    columns='ul#test#6248_classhot#last#functional_bin',
    values='count',
    fill_value=0
).reset_index()
# class test time
lnl_class_time = df3.groupby(['mm_code', 'sub_unit_fab_lot', 'sub_unit_wafer_id'])['out_date'].agg('max').reset_index()
lnl_join_bin = pd.merge(lnl_ib, lnl_fb, on=['sub_unit_fab_lot', 'sub_unit_wafer_id'], how='inner')
lnl_join = pd.merge(lnl_class_time, lnl_join_bin, on=['sub_unit_fab_lot', 'sub_unit_wafer_id'], how='inner')

# 2 other product, class test step 7820
df = pd.concat([df1, df2, df4, df5, df6, df7, df8, df10, df11, df12,], ignore_index=True)
df = df.drop_duplicates()
# IB
other_grouped_ib = df.groupby(['sub_unit_fab_lot', 'sub_unit_wafer_id', 'ul#test#7820_classhot#last#interface_bin']).size().reset_index(name='count')
other_grouped_ib['ul#test#7820_classhot#last#interface_bin'] = other_grouped_ib['ul#test#7820_classhot#last#interface_bin'].astype(str)
other_ib = other_grouped_ib.pivot_table(
    index=['sub_unit_fab_lot', 'sub_unit_wafer_id'],
    columns='ul#test#7820_classhot#last#interface_bin',
    values='count',
    fill_value=0
).reset_index()
# FB
other_grouped_fb = df.groupby(['sub_unit_fab_lot', 'sub_unit_wafer_id', 'ul#test#7820_classhot#last#functional_bin']).size().reset_index(name='count')
other_grouped_fb['ul#test#7820_classhot#last#functional_bin'] = other_grouped_fb['ul#test#7820_classhot#last#functional_bin'].astype(str)
other_fb = other_grouped_fb.pivot_table(
    index=['sub_unit_fab_lot', 'sub_unit_wafer_id'],
    columns='ul#test#7820_classhot#last#functional_bin',
    values='count',
    fill_value=0
).reset_index()

# class test time
other_class_time = df.groupby(['mm_code', 'sub_unit_fab_lot', 'sub_unit_wafer_id'])['out_date'].agg('max').reset_index()
other_join_bin = pd.merge(other_ib, other_fb, on=['sub_unit_fab_lot', 'sub_unit_wafer_id'], how='inner')
other_join = pd.merge(other_class_time, other_join_bin, on=['sub_unit_fab_lot', 'sub_unit_wafer_id'], how='inner')
wafer_class = pd.concat([lnl_join, other_join], ignore_index=True)

# get all the ib columns
ib_columns = [col for col in wafer_class.columns if len(col) < 5]
print(ib_columns)
# print(wafer_class.columns)
# calculate all class test units for each wafer
wafer_class['class_tested_units'] = wafer_class[ib_columns].sum(axis=1)
# calculate sbb failure bins for each wafer
wafer_class['sbb_class'] = wafer_class[['8.0', '10.0', '36.0', '99.0']].sum(axis=1)
# calculate Si remaining combine bin
# Define conditions
conditions = [
    wafer_class['mm_code'] == 'METEOR LAKE-BASE-P682',
    wafer_class['mm_code'] == 'METEOR LAKE-BASE-P281',
    wafer_class['mm_code'] == 'ARROW LAKE-BASE-P281',
    wafer_class['mm_code'] == "ARROW LAKE-BASE-H682",
    wafer_class['mm_code'] == "LUNAR LAKE-BASE-M442",
    wafer_class['mm_code'] == "ARROW LAKE-BASE-S816"
]

# Define choices based on conditions
choices = [
    wafer_class['902.0'] + wafer_class['950.0']+ wafer_class['9768.0'],  # Condition 1
    wafer_class['902.0'] + wafer_class['950.0']+ wafer_class['9768.0'],  # Condition 2
    wafer_class['902.0'] + wafer_class['9768.0']                # Condition 3
]

case4=wafer_class['902.0'] + wafer_class['950.0'] + wafer_class['9768.0']
for each in wafer_class.columns:
    if each.isdigit() and (float(each)>=1590.0 and float(each)<=1599.0):
        case4 += wafer_class[each]
choices.append(case4)


case5 = wafer_class['900.0'] + wafer_class['9722.0'] + wafer_class['9723.0']+ wafer_class['9735.0']+ wafer_class['9734.0']             # Condition 5 (example of a different operation)
choices.append(case5)


case6 = wafer_class['912.0'] + wafer_class['9768.0']
for each in wafer_class.columns:
    if each.isdigit() and (float(each)>=1590.0 and float(each)<=1599.0):
        case6 += wafer_class[each]
choices.append(case6)


# Use np.select to apply conditions and choices
wafer_class['Si_class'] = np.select(conditions, choices, default=np.nan)
# get percentage
wafer_class['sbb_class%'] = wafer_class['sbb_class']/wafer_class['class_tested_units']*100
wafer_class['Si_class%'] = wafer_class['Si_class']/wafer_class['class_tested_units']*100
wafer_class.to_csv("wafer_class.csv", index = False)

#wafer_class = pd.read_csv("wafer_class_0507.csv")
# load process data
df_process = pd.read_csv("0717_process.csv")
# ACW has 2 process steps: 852060, 216365, combine into single column
df_process['ACW_time'] = df_process['END_TIME@ENTITY@NTSC@TCW@852060'].fillna(df_process['END_TIME@ENTITY@NTSC@TCW@216365'])
df_process['ACW_site'] = df_process['SITE@NTSC@TCW@852060'].fillna(df_process['SITE@NTSC@TCW@216365'])
df_process['ACW_entity'] = df_process['ENTITY@NTSC@TCW@852060'].fillna(df_process['ENTITY@NTSC@TCW@216365'])
df_process['ACW_chamber'] = df_process['CHAMBER@NTSC@Process-1@TCW@852060'].fillna(df_process['CHAMBER@NTSC@Process-1@TCW@216365'])
# calulate SDHMT yield
ib_columns_list = [col for col in df_process.columns if col.startswith('IB')]
print(ib_columns_list)
use_column_list = ib_columns_list + ['FB693@SDT@WAFER', 'FB9301@SDT@WAFER']
df_process[use_column_list] = df_process[use_column_list].fillna(0)
df_process['sdhmt_tested_units'] = df_process[ib_columns_list].sum(axis=1) - df_process['FB693@SDT@WAFER'] - df_process['FB9301@SDT@WAFER']
df_process['sdhmt_yield%'] = df_process['FB101@SDT@WAFER']/df_process['sdhmt_tested_units']*100
df_process['sbb_sdhmt'] = df_process[['IB8@SDT@WAFER', 'IB10@SDT@WAFER', 'IB36@SDT@WAFER','IB99@SDT@WAFER']].sum(axis=1)
df_process['sbb_sdhmt%'] = df_process['sbb_sdhmt']/df_process['sdhmt_tested_units']*100
df_process['Si_sdhmt'] = df_process[['IB9@SDT@WAFER', 'IB97@SDT@WAFER']].sum(axis=1)
df_process['Si_sdhmt%'] = df_process['Si_sdhmt']/df_process['sdhmt_tested_units']*100
df_process.to_csv("0613_df_process_calculated.csv", index = False)

# filter only the key columns
#key_columns = ['LOT', 'WAFER', 'WAFER_ID', 'PRODUCT@STARTS', 'LOT_TYPE@STARTS', 'sdhmt_tested_units', 'sdhmt_yield%', 'sbb_sdhmt', 'sbb_sdhmt%', 'Si_sdhmt','Si_sdhmt%', 'ACW_time', 'ACW_site', 'ACW_entity', 'ACW_chamber', 'ENTITY@NTSC@AME@231368', 'END_TIME@ENTITY@NTSC@AME@231368', 'SITE@NTSC@AME@231368', 'FDC_SUMMARY@Mean@InnerHeliumLeak@S4_BT@MFGxx_TSV_AME_CE_RVL@AME@231368']
key_columns = ['PRODUCT@STARTS', 'LOT', 'WAFER', 'WAFER_ID','sdhmt_tested_units', 'sdhmt_yield%', 'sbb_sdhmt', 'sbb_sdhmt%', 'Si_sdhmt','Si_sdhmt%', 'ACW_time', 'ACW_site', 'ACW_entity', 'ACW_chamber', 'ENTITY@NTSC@AME@231368', 'END_TIME@ENTITY@NTSC@AME@231368', 'SITE@NTSC@AME@231368', 'FDC_SUMMARY@Mean@InnerHeliumLeak@S4_BT@MFGxx_TSV_AME_CE_RVL@AME@231368', 'TEST_END_DATE@SORT@WAFER', 'PRODUCT@SORT@WAFER']
df_process_filter = df_process[key_columns]

# merge with class test data
df_join_class = pd.merge(df_process_filter, wafer_class, left_on ='WAFER_ID', right_on='sub_unit_wafer_id', how='left')
df_join_class.to_csv("df_join_class.csv", index = False)