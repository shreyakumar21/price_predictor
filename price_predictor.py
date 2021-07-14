import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('iphone_price.csv') #add your csv file
plt.scatter(data['version'], data['price'])
plt.show()
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[20]]))
