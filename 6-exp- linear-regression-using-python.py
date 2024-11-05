'''Implement linear regression using python 

=================================
Explanation:
=================================

===> To run this program you need to install the pandas Module

---> pandas Module is used to read csv files

===> To install, Open Command propmt and then execute the following command

---> pip install pandas


And, then you need to install the matplotlib Module 

---> matplotlib Module is used to plot the graphs

===> To install, Open Command propmt and then execute the following command

---> pip install matplotlib

Finally, you need to create dataset called "Age_Income.csv" file.

===============================
Source Code :
===============================

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# To read data from Age_Income.csv file
dataFrame = pd.read_csv('Age_Income.csv')
# To place data in to age and income vectors
age = dataFrame['Age']
income = dataFrame['Income']

# number of points
num = np.size(age)
# To find the mean of age and income vector
mean_age = np.mean(age)
mean_income = np.mean(income)

# calculating cross-deviation and deviation about age
CD_ageincome = np.sum(income*age) - num*mean_income*mean_age
CD_ageage = np.sum(age*age) - num*mean_age*mean_age

# calculating regression coefficients
b1 = CD_ageincome / CD_ageage
b0 = mean_income - b1*mean_age
# to display coefficients
print("Estimated Coefficients :")
print("b0 = ",b0,"\nb1 = ",b1)
# To plot the actual points as scatter plot
plt.scatter(age, income, color = "b",marker = "o")
# TO predict response vector
response_Vec = b0 + b1*age
# To plot the regression line
plt.plot(age, response_Vec, color = "r")
# Placing labels
plt.xlabel('Age')
plt.ylabel('Income')
# To display plot
plt.show()


Age_Income.csv
Age,Income
25,25000
23,22000
24,26000
28,29000
34,38600
32,36500
42,41000
55,81000
45,47500
