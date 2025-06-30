import numpy as np
import matplotlib.pyplot as plt

# Use the same style as DeepLearning.AI courses
#plt.style.use('./deeplearning.mplstyle')  # Ensure you have this style or comment this line

# 1. Training Data
x_train = np.array([1.0, 2.0])  # Size in 1000 sqft
y_train = np.array([300.0, 500.0])  # Price in 1000s of dollars

# Print training data
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# 2. Number of training examples
m = x_train.shape[0]
print(f"Number of training examples: {m}")

# 3. Access a training example
i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# 4. Plot the training data
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.show()

# 5. Model function: f_wb = w * x + b
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    return w * x + b

# 6. Try different model parameters
w = 200
b = 100
print(f"Chosen parameters -> w: {w}, b: {b}")

# Compute model predictions
tmp_f_wb = compute_model_output(x_train, w, b)

# Plot predictions vs. actual values
plt.plot(x_train, tmp_f_wb, c='b', label='Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual')
plt.title("Housing Prices - Model vs Data")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")
plt.legend()
plt.show()

# 7. Predict price for a house with 1200 sqft (x = 1.2)
x_input = 1.2
predicted_price = w * x_input + b
print(f"Predicted price for 1200 sqft house: ${predicted_price:.0f} thousand dollars")
