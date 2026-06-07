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
    )

#solver='sgd',                # SGD is often more stable for pure online learning than Adam
        #learning_rate='adaptive',    # Automatically adjusts the step size if MSE spikes
        #learning_rate_init=0.01,     # 10x larger starting step to react faster
        #random_state=42