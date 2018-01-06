import pandas as pd
import os.path

try:
    with open('pickle_df/clean_df.pkl', 'rb') as pickle_file:
        loans = pd.read_pickle(pickle_file)
except:
    print("Please make sure to run cleaning_data.py first")

#Drop columns that have more than 0.01% missing information
high_null_column_count = ['emp_title', 'emp_length', 'pub_rec_bankruptcies']
loans = loans.drop(high_null_column_count, axis=1)

#drop rows with any remaining null values
loans = loans.dropna(axis=0)

#Drop rows which contain only unique values
cols = loans.columns
unique_value_columns = []
for col in cols:
    col_series = loans[col].dropna().unique()
    if len(col_series) == 1:
        unique_value_columns.append(col)
loans = loans.drop(unique_value_columns, axis=1)

#Drop these columns since they require extra feature engineering
loans = loans.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)
#Convert these for easier processing
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")

cat_columns = ["home_ownership", "verification_status", "purpose", "term"]

#Use dummy variables to convert categorical to numerical
dummy_df = pd.get_dummies(loans[cat_columns])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cat_columns, axis=1)

loans.to_pickle('pickle_df/prepared_df.pkl')
