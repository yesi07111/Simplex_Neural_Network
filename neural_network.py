import numpy as np

class LinearNeuralNetwork:
    """
    Descripción:
    ------------
    Clase que representa una red neuronal de alimentación hacia adelante simple con funciones de activación lineales.

    Atributos:
    ----------
    input_size : int
        Tamaño de la capa de entrada.
    hidden_size : int
        Tamaño de la capa oculta.
    output_size : int
        Tamaño de la capa de salida.

    Métodos:
    --------
    forward(X: np.ndarray) -> np.ndarray
        Realiza la propagación hacia adelante de la red neuronal.
    linear_activation(x: np.ndarray) -> np.ndarray
        Calcula la función de activación lineal de la entrada.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Descripción:
        ------------
        Inicializa la red neuronal con pesos y sesgos aleatorios.

        Parámetros:
        -----------
        input_size : int
            Tamaño de la capa de entrada.
        hidden_size : int
            Tamaño de la capa oculta.
        output_size : int
            Tamaño de la capa de salida.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicializar pesos y sesgos con valores aleatorios más pequeños
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros(output_size)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Descripción:
        ------------
        Realiza la propagación hacia adelante de la red neuronal.

        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrada para la red neuronal.

        Retorna:
        --------
        np.ndarray
            Salida de la red neuronal después de la propagación hacia adelante.
        """
        # Calcular la activación de la capa oculta
        hidden_layer = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_activation = self.linear_activation(hidden_layer)

        # Calcular la activación de la capa de salida
        output_layer = (
            np.dot(hidden_layer_activation, self.weights_hidden_output)
            + self.bias_output
        )
        output_layer_activation = self.linear_activation(output_layer)

        return output_layer_activation

    @staticmethod
    def linear_activation(x: np.ndarray) -> np.ndarray:
        """
        Descripción:
        ------------
        Calcula la función de activación lineal de la entrada.

        Parámetros:
        -----------
        x : np.ndarray
            Entrada para la función de activación lineal.

        Retorna:
        --------
        np.ndarray
            Resultado de aplicar la función de activación lineal a la entrada.
        """
        return x

    def linear_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Descripción:
        ------------
        Calcula el error absoluto medio entre los valores verdaderos y las predicciones.

        Parámetros:
        -----------
        y_true : np.ndarray
            Valores verdaderos de las etiquetas.
        y_pred : np.ndarray
            Valores predichos por el modelo.

        Retorna:
        --------
        float
            El error absoluto medio entre los valores verdaderos y las predicciones.
        """
        return np.sum(np.abs(y_true - y_pred))