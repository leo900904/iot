from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Function to generate plot
def generate_plot(n, variance, a, b, c):
    # Generate random data points
    x = np.random.uniform(-10, 10, n)
    y = a * x + b + c * np.random.normal(0, variance, n)

    # Reshape x to be a 2D array for scikit-learn
    x = x.reshape(-1, 1)

    # Use scikit-learn for linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label=f'Regression line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    plt.title('Linear Regression Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Convert plot to PNG image and encode as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# API route to update plot
@app.route('/update_plot', methods=['POST'])
def update_plot():
    n = int(request.form.get('n', 100))
    variance = float(request.form.get('variance', 5))
    a = float(request.form.get('a', np.random.uniform(-10, 10)))
    b = float(request.form.get('b', 50))
    c = float(request.form.get('c', np.random.uniform(0, 100)))

    plot_url = generate_plot(n, variance, a, b, c)
    
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
