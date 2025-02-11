import numpy as np
from neural_network import LinearNeuralNetwork

def personalized_simplex_optimizer(nn: LinearNeuralNetwork, X: np.ndarray, y: np.ndarray, max_iter=5000, tol=1e-6):
    """
    Optimiza la red neuronal utilizando una implementación personalizada del algoritmo Simplex clásico para ajustar los pesos de una red neuronal lineal.
    El optimizador busca minimizar el error cuadrático medio (MSE) entre las predicciones de la red neuronal y los valores reales.
    Se inicializa un conjunto de soluciones (simplex) y se iteran operaciones de reflexión, expansión, y contracción para encontrar el conjunto óptimo de parámetros.

    """
    # Calcular la media de los datos de entrada
    X_mean = np.mean(X, axis=0)
    # Calcular la desviación estándar de los datos de entrada
    X_std = np.std(X, axis=0)
    # Normalizar los datos de entrada
    X_normalized = (X - X_mean) / X_std

    # Concatenar todos los parámetros de la red neuronal en un solo vector
    initial_params = np.concatenate([
        nn.weights_input_hidden.flatten(),
        nn.bias_hidden,
        nn.weights_hidden_output.flatten(),
        nn.bias_output,
    ])
    # Obtener el número total de parámetros
    n_params = len(initial_params)
    # Inicializar el simplex con ceros
    simplex = np.zeros((n_params + 1, n_params))
    # Asignar los parámetros iniciales al primer vértice del simplex
    simplex[0] = initial_params

    # Crear un simplex inicial con mayor variabilidad
    for i in range(1, n_params + 1):
        # Añadir ruido aleatorio a los parámetros iniciales para crear otros vértices del simplex
        simplex[i] = initial_params + np.random.randn(n_params) * 0.1

    def linear_error_function(params):
        # Asignar los parámetros a la red neuronal
        input_size, hidden_size, output_size = nn.input_size, nn.hidden_size, nn.output_size
        nn.weights_input_hidden = params[: input_size * hidden_size].reshape(input_size, hidden_size)
        nn.bias_hidden = params[input_size * hidden_size : input_size * hidden_size + hidden_size]
        nn.weights_hidden_output = params[input_size * hidden_size + hidden_size : -output_size].reshape(hidden_size, output_size)
        nn.bias_output = params[-output_size:]

        # Realizar la predicción con la red neuronal
        y_pred = nn.forward(X_normalized)
        # Calcular el error cuadrático medio (MSE)
        return np.mean((y - y_pred) ** 2)

    # Evaluar el error inicial para cada vértice del simplex
    errors = np.apply_along_axis(linear_error_function, 1, simplex)

    for iteration in range(max_iter):
        # Ordenar los vértices del simplex por el error
        indices = np.argsort(errors)
        simplex = simplex[indices]
        errors = errors[indices]

        # Verificar si la desviación estándar de los errores es menor que la tolerancia
        if np.std(errors) < tol:
            break

        # Calcular el centroide del simplex sin el peor vértice
        centroid = np.mean(simplex[:-1], axis=0)

        # Realizar la reflexión del peor vértice
        reflection = centroid + (centroid - simplex[-1])
        reflection_error = linear_error_function(reflection)

        if reflection_error < errors[0]:
            # Realizar la expansión si la reflexión es mejor que el mejor vértice
            expansion = centroid + 2 * (reflection - centroid)
            expansion_error = linear_error_function(expansion)
            if expansion_error < reflection_error:
                simplex[-1] = expansion
                errors[-1] = expansion_error
            else:
                simplex[-1] = reflection
                errors[-1] = reflection_error
        elif reflection_error < errors[-2]:
            # Aceptar la reflexión si es mejor que el segundo peor vértice
            simplex[-1] = reflection
            errors[-1] = reflection_error
        else:
            # Realizar la contracción si la reflexión no es suficientemente buena
            contraction = centroid + 0.5 * (simplex[-1] - centroid)
            contraction_error = linear_error_function(contraction)
            if contraction_error < errors[-1]:
                simplex[-1] = contraction
                errors[-1] = contraction_error
            else:
                # Reducción adaptativa si la contracción no mejora
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                errors = np.apply_along_axis(linear_error_function, 1, simplex)

                # Perturbación aleatoria si el progreso es lento
                if iteration % 1000 == 0:
                    simplex += np.random.randn(*simplex.shape) * 0.01 + np.random.random_integers(-3, 3) * np.random.random_integers(100, 10000)

    # Asignar los parámetros optimizados a la red neuronal
    optimized_params = simplex[0]
    nn.weights_input_hidden = optimized_params[: nn.input_size * nn.hidden_size].reshape(nn.input_size, nn.hidden_size)
    nn.bias_hidden = optimized_params[nn.input_size * nn.hidden_size : nn.input_size * nn.hidden_size + nn.hidden_size]
    nn.weights_hidden_output = optimized_params[nn.input_size * nn.hidden_size + nn.hidden_size : -nn.output_size].reshape(nn.hidden_size, nn.output_size)
    nn.bias_output = optimized_params[-nn.output_size:]

    # Realizar la predicción final con los parámetros optimizados
    y_pred = nn.forward(X_normalized)
    # Calcular el error cuadrático medio final
    final_loss = np.mean((y - y_pred) ** 2)
    return y_pred, final_loss