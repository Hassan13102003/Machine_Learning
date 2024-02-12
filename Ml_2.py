# Using Gradient Descent Algorithm Find the minima of a uni-variate function:
# 		 f(x)= x e^(-x^2),   -√1.5 ≤ x ≤ 0.

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * np.exp(-x ** 2)


def df(x):
    return (1 - 2 * x ** 2) * np.exp(-x ** 2)


def gradient_descent(starting_point, learning_rate, stopping_threshold):
    x = starting_point
    iterations = 0

    while True:
        gradient = df(x)
        x = x - learning_rate * gradient
        iterations += 1

        if np.linalg.norm(gradient) < stopping_threshold:
            break

    return x, iterations, f(x)


# Define the range for x
x_range = np.linspace(-np.sqrt(1.5), 0, 100)

# Choose a starting point and stopping threshold
initial_x = -np.sqrt(1.5)
stopping_threshold = 0.001

# Different step size values
step_sizes = [0.005, 0.01, 0.05]

# Prepare the table
print("{:<10} {:<20} {:<20} {:<20}".format("Step Size", "Iterations", "Minima Value", "Function Value"))

for step_size in step_sizes:
    minima_x, num_iterations, minima_value = gradient_descent(initial_x, step_size, stopping_threshold)
    function_value_at_minima = f(minima_x)

    print("{:<10} {:<20} {:<20} {:<20}".format(step_size, num_iterations, minima_value, function_value_at_minima))

    # Plot the function for each step size
    plt.plot(x_range, f(x_range), label=r'$f(x) = x e^{-x^2}$')
    plt.scatter(minima_x, function_value_at_minima, color='red', marker='o', label='Minima')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gradient Descent for Step Size {step_size}')
    plt.legend()
    plt.show()
