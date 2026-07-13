import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
# ✅ use pyplot, not plt directly

# ✅ use pd.read_csv() with parentheses and dot
dataset = pd.read_csv(r"C:\Users\DURGA\Downloads\dataset.csv")

# Example: check first 5 rows
print(dataset.head())

# Example plot (if dataset has numeric columns)
plt.plot(dataset.iloc[:,0], dataset.iloc[:,1])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Sample Plot")
plt.show()



