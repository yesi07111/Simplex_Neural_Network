import numpy as np
from scipy.optimize import linprog
from neural_network import LinearNeuralNetwork

def optimize_nn(nn: LinearNeuralNetwork, X: np.ndarray, y: np.ndarray):
    """
    Descripción:
    ------------
    Optimiza la red neuronal utilizando el algoritmo de optimización lineal Simplex.

    Parámetros:
    -----------
    nn : LinearNeuralNetwork
        Instancia de la red neuronal a optimizar.
    X : np.ndarray
        Datos de entrada para el entrenamiento.
    y : np.ndarray
        Valores verdaderos de las etiquetas para el entrenamiento.

    Retorna:
    --------
    Tuple[np.ndarray, float]
        Predicciones de la red neuronal optimizada y la pérdida final.
    """
    # Concatenar todos los parámetros de la red neuronal
    initial_params = np.concatenate([
        nn.weights_input_hidden.flatten(),
        nn.bias_hidden,
        nn.weights_hidden_output.flatten(),
        nn.bias_output,
    ])

    # Definir la función de error para minimizar
    def linear_error_function(params):
        input_size, hidden_size, output_size = nn.input_size, nn.hidden_size, nn.output_size
        nn.weights_input_hidden = params[: input_size * hidden_size].reshape(input_size, hidden_size)
        nn.bias_hidden = params[input_size * hidden_size : input_size * hidden_size + hidden_size]
        nn.weights_hidden_output = params[input_size * hidden_size + hidden_size : -output_size].reshape(hidden_size, output_size)
        nn.bias_output = params[-output_size:]

        y_pred = nn.forward(X)
        return nn.linear_error(y, y_pred)

    # Crear restricciones para asegurar que los parámetros no sean triviales
    bounds = [(None, None) for _ in range(len(initial_params))]

    # Definir la función objetivo lineal
    c = np.random.rand(len(initial_params))  # Coeficientes aleatorios para la función objetivo

    # Aplicar el algoritmo Simplex
    result = linprog(c, A_eq=None, b_eq=None, bounds=bounds, method='simplex')

    # Verificar si la optimización fue exitosa
    if result.success:
        optimized_params = result.x
        input_size, hidden_size, output_size = nn.input_size, nn.hidden_size, nn.output_size
        nn.weights_input_hidden = optimized_params[: input_size * hidden_size].reshape(input_size, hidden_size)
        nn.bias_hidden = optimized_params[input_size * hidden_size : input_size * hidden_size + hidden_size]
        nn.weights_hidden_output = optimized_params[input_size * hidden_size + hidden_size : -output_size].reshape(hidden_size, output_size)
        nn.bias_output = optimized_params[-output_size:]
    else:
        print("La optimización no fue exitosa. Usando parámetros iniciales.")

    y_pred = nn.forward(X)
    final_loss = nn.linear_error(y, y_pred)
    return y_pred, final_loss