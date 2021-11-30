import pandas as pd

leak = pd.read_csv('.csv files/test_leakage.csv', index_col=0)
pred = pd.read_csv('.csv files/test_leakage_prediction.csv', index_col=0)

print((abs(leak - pred)/100).mean())
