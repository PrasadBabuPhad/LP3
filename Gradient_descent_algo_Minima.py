
# Gradient Descent Algorithm

# Define the function
def function(x):
    return (x + 3) ** 2

# Derivative of the function
def derivative(x):
    return 2 * (x + 3)

# Initial values
x = 2
learning_rate = 0.1
iterations = 50

print("Initial x:", x)

# Gradient Descent
for i in range(iterations):
    gradient = derivative(x)
    x = x - learning_rate * gradient

    print("Iteration", i + 1, "x =", round(x, 6),
          "y =", round(function(x), 6))

print("\nLocal Minimum:")
print("x =", round(x, 6))
print("y =", round(function(x), 6))

