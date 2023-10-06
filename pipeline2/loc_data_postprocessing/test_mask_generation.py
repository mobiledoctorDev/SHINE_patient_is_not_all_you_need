import pandas as pd
import numpy as np
from mask_and_normalize import get_mask_sr, normalize

# Test mask generation with sample data
# create sample dataframe
df = pd.DataFrame({
    'age': [25, 30, np.nan, 0, 999999, 0, 1, 2, 3, 4, 5],
})
# create mask column
df['mask'] = get_mask_sr(df['age'])
df['norm'] = normalize(df['age'], save_params=False, save_figure=False, display=False)
# print dataframe
print(df)
