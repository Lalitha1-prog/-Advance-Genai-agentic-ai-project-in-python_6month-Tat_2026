import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Salary_Data = pd.read_csv(r"C:\Users\DURGA\Downloads\Salary_Data.csv")
X = Salary_Data.iloc[:, :-1].values
y = Salary_Data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_coef= regressor.coef_
print(m_coef)

c_intercept=regressor.intercept_
print(c_intercept)

y_12=m_coef * 12 + c_intercept
print(y_12)

y_20=m_coef * 20 + c_intercept
print(y_20)

bias_score=regressor.score(X_train,y_train)
print(bias_score)

variance_score=regressor.score(X_test,y_test)
print(variance_score)

#statistics integration
Salary_Data.mean()
Salary_Data['Salary'].mean()
Salary_Data['YearsExperience'].mean()

Salary_Data.median()
Salary_Data['Salary'].median()
Salary_Data['YearsExperience'].median()

Salary_Data.var()
Salary_Data['Salary'].var()
Salary_Data['YearsExperience'].var()

Salary_Data.std()
Salary_Data['Salary'].std()
Salary_Data['YearsExperience'].std()

from scipy.stats import variation
variation(Salary_Data.values)
variation(Salary_Data['Salary'])
variation(Salary_Data['YearsExperience'])

Salary_Data.corr()

Salary_Data['Salary'].corr(Salary_Data['YearsExperience'])

# Skewness
Salary_Data.skew()
Salary_Data['Salary'].skew()
Salary_Data.sem()

# Z-score
import scipy.stats as stats
Salary_Data.apply(stats.zscore)
stats.zscore(Salary_Data['Salary'])

# ANOVA # SSR,SSE,SST
y_mean=np.mean(y)
SSR= np.sum((y_pred-y_mean)**2)
print(SSR)

y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

mean_total=np.mean(Salary_Data.values)
SST=np.sum((Salary_Data.values-mean_total)**2)
print(SST)

r_square=1-(SSR/SST)
r_square

print(r_square)
print(bias_score)
print(variance_score)



