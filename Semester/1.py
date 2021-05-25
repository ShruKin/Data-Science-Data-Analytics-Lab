import pandas as pd
import numpy as np

d = {'f1': [1, 5, 0], 'f2': [2, 6, 6], 'f3': [
    3, np.nan, 9], 'f4': [4, 7, np.nan]}
df = pd.DataFrame.from_dict(d)
# df.head()

df['f3'].fillna(value=df["f3"].mean(), inplace=True)
df['f4'].fillna(value=df["f4"].mean(), inplace=True)

print(df.head())
