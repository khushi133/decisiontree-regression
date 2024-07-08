import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import streamlit as st

# Set the title of the Streamlit app
st.title("Interactive Decision Tree Regression")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# User input for test size
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# User input for decision tree parameters
max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)

# Generate dataset
X = np.linspace(-5, 5, 100)
y = np.exp(X)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Plot overall dataset
st.subheader("Overall Dataset Display")
fig, ax = plt.subplots()
ax.scatter(X, y, edgecolors='k', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Overall DataSet Display')
st.pyplot(fig)

# Plot training and testing dataset
st.subheader("Training and Testing Dataset Display")
fig, ax = plt.subplots()
ax.scatter(x_train, y_train, edgecolors='g', marker='*', label="Training_dataset")
ax.scatter(x_test, y_test, edgecolors='r', marker='^', label="Testing_dataset")
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Training and Testing DataSet Display')
st.pyplot(fig)

# Fitting Decision Tree Regression to the dataset
clf = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
clf = clf.fit(x_train.reshape(-1, 1), y_train)

# Plot decision tree
st.subheader("Decision Tree Plot")
fig, ax = plt.subplots(figsize=(12, 6))
tree.plot_tree(clf, filled=True, ax=ax)
st.pyplot(fig)

# Predicting the test set results
y_pred = clf.predict(x_test.reshape(-1, 1))

st.subheader("Predicted vs True Values")
st.write("Predicted values:", y_pred)
st.write("True values:", y_test)

# RMSE for testing dataset
rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
st.write('RMSE for Testing Decision Tree Regressor:', rmse_test)

# RMSE for training dataset
rmse_train = np.sqrt(metrics.mean_squared_error(y_train, clf.predict(x_train.reshape(-1, 1))))
st.write('RMSE for Training Decision Tree Regressor:', rmse_train)

# Plot of testing and predicted dataset points
st.subheader("Testing and Predicted Dataset Display")
fig, ax = plt.subplots()
ax.scatter(x_test, y_test, marker='o', label="Testing_dataset")
ax.scatter(x_test, y_pred, marker='^', edgecolors='r', label="Predicted_dataset")
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Testing and Predicted Dataset Display')
st.pyplot(fig)
