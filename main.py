import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

# Read the dataset named 'reG_veri':
df = pd.read_csv('Data/reG_veri.csv')  # Assuming the dataset is in CSV format and in the same directory

# Identify independent and dependent variables:
# Assumption: Independent variables are 'X1', 'X2', ..., 'Xn' and the dependent variable is 'Fiyat'
independent_variables = df.columns[:-1]  # All columns except the last one
X = df[independent_variables]
y = df['Fiyat']

# Normalize the independent (X) variables with Min-Max scaling:
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X = pd.DataFrame(X_normalized, columns=independent_variables)

# Split the data into training and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Create and train a linear regression model using Scikit-learn:
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate mean absolute error (MAE) on training data predictions:
y_pred_train = model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

# Create a DataFrame containing the coefficients and MAE value of the model:
training_summary = pd.DataFrame(model.coef_, index=independent_variables, columns=['Coefficients'])
training_summary.loc['Training MAE'] = mae_train

# Calculate test error by making predictions on test data:
y_pred_test = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

# Create a DataFrame containing the coefficients and test MAE of the model:
test_summary = pd.DataFrame(model.coef_, index=independent_variables, columns=['Coefficients'])
test_summary.loc['Test MAE'] = mae_test

# Print the results:
print("Training Data Model Summary:\n", training_summary)
print("\nTest Data Model Summary:\n", test_summary)


############################################

# Add a constant to the independent variables:
X_sm = sm.add_constant(X)

# Create a linear regression model with Statsmodels:
model_sm = sm.OLS(y, X_sm).fit()

# Print the model results:
model_results = model_sm.summary()
print('Statsmodels:', model_results)