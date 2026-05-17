import warnings
from sklearn.neural_network import MLPRegressor

def create_online_model():
    """
    Creates MLPRegressor neural network for online learning purpose (warm_start=True).

    One hidden layer with 10 neurons, each of them represents 1 positional coordinate of Rabbit movement.
    """
    
    # Filter convergence (zbieżność) warnings
    warnings.filterwarnings("ignore")
    
    return MLPRegressor(
        hidden_layer_sizes=(10,), 
        max_iter=1, 
        warm_start=True, 
        random_state=42
    )