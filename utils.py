import numpy as np
import matplotlib.pyplot as plt
from model import create_online_model

def run_simulation(rabbit, wolf, model, n_steps):
    '''
    
    '''

    path_x, path_y = [], []
    preds_x, preds_y = [], []
    mse_history = []

    # Keeps check of first weights being initialized
    is_model_initialized = False

    for i in range(n_steps):
        
        # 1. Next Rabbit move
        pos = rabbit.move()
        path_x.append(pos[0])
        path_y.append(pos[1])

        # Moving window of 5 Rabbit positions
        X_window = wolf.prepare_input_data()

        # Updating must be done first to avoid losing data before parsing it (Data Leakage)
        wolf.update_history(pos)

        if X_window is None:
            continue

        y_actual = pos.reshape(1,-1)

        # 2. First weights initialization 
        if not is_model_initialized:
            model.partial_fit(X_window, y_actual)
            is_model_initialized = True

        # 3. Wolf makes a prediction
        pos_pred = model.predict(X_window).flatten()
        preds_x.append(pos_pred[0])
        preds_y.append(pos_pred[1])
        
        # 4. Calculate current MSE and append history
        mse = float(np.mean((y_actual.flatten() - pos_pred) ** 2))
        mse_history.append(mse)
        
        # 5. Model reweights
        model.partial_fit(X_window, y_actual)
        
    return path_x, path_y, preds_x, preds_y, mse_history


def generate_report(path_x, path_y, preds_x, preds_y, mse_history):
    '''
    Generates a visualization of model work:
    1 - Rabbit path traversal
    2 - log(MSE) over learing time, scale better shows the decrease in error
    3 - Shows Rabbit and Wolf paths to visualize who model adapted to dynamic Rabbit movements
    '''
    
    plt.figure(figsize=(12, 8))

    # 1. Rabbit path
    plt.subplot(2, 1, 1)
    plt.plot(path_x, path_y, 'g-', label='Trasa królika')
    plt.scatter(path_x[0], path_y[0], c='blue', label='Start')
    plt.title("Symulacja trajektorii i predykcji")
    plt.legend()
    plt.grid(True)

    # 2. MSE convergence
    plt.subplot(2, 2, 3)
    plt.plot(mse_history, 'r')
    plt.yscale('log') 
    plt.title("Błąd MSE w czasie")
    plt.xlabel("Krok")
    plt.ylabel("Log(MSE)")

    # 3. Wolf and Rabbit path comparison
    plt.subplot(2, 2, 4)
    plt.plot(path_x, path_y, 'b-', label='Realna droga', linewidth=3, alpha=0.5)
    plt.plot(preds_x, preds_y, 'o-', label='Predykcja', linewidth=2)
    plt.title("Predykcja vs Rzeczywistość")
    plt.xlabel("Oś X")
    plt.ylabel("Oś Y")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()