import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Example data
data = {'Rater1': ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'yes', 'no'],
        'Rater2': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'yes', 'no']}

df = pd.DataFrame(data)

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(df['Rater1'], df['Rater2'])

print(f"Cohen's Kappa: {kappa}")
