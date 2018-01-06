import pandas as pd

loans = pd.read_csv('lending_data/LoanStats3a.csv', skiprows=1, low_memory=False)
loans.drop_duplicates()
half_count = len(loans) / 2
#If null values are greater than half the amount, drop those columns
loans = loans.dropna(thresh=half_count, axis=1)
#Long text explanation of the loan can't be used

# We need to get rid of all these columns that leak data about the future of this person getting a loan
data_leakage_cols = ['disbursement_method', 'funded_amnt', 'funded_amnt_inv', 'issue_d', 'out_prncp', 'total_pymnt', 'out_prncp_inv',
    'total_pymnt_inv','total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt']

# These are either redundant or too hard to process
redundant_cols = ['grade', 'sub_grade', 'zip_code', 'hardship_flag', 'debt_settlement_flag', 'desc']


loans = loans.drop(data_leakage_cols, axis=1)
loans = loans.drop(redundant_cols, axis=1)

#find and remove columns which are only unique
cols = loans.columns
unique_value_columns = []
for col in cols:
    col_series = loans[col].dropna().unique()
    if len(col_series) == 1:
        unique_value_columns.append(col)
loans = loans.drop(unique_value_columns, axis=1)
print(loans['acc_now_delinq'].unique())
print(loans['delinq_amnt'].unique())
print(loans['tax_liens'].unique())

#TODO one-hot encode emp title

# Turn loan_status into a binary category of paid off or charged off
mapping_dict = {
    "loan_status" : {
    "Fully Paid" : 1,
    'Charged Off' : 0
    }
}
loans = loans[(loans['loan_status'] == "Fully Paid") | (loans['loan_status'] == "Charged Off")]

loans = loans.replace(mapping_dict)

loans.to_pickle('pickle_df/clean_df.pkl')
