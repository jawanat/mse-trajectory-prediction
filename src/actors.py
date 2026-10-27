import numpy as np
import matplotlib.pyplot as plt

class Rabbit:
    '''
    Generates Rabbit coordinates data with constant speed.
    Uses velocity * 2D rotation matrix and added noise from N(0,1) to create a dynamic movement path for Wolf to follow.
    '''
    
    def __init__(self, start_pos=(0, 0), speed=0.5, random_state=None):
        self.pos = np.array(start_pos, dtype=float)
        self.velocity = np.array([1.0, 0.0]) * speed    # acceleration vector
        self.speed = speed
        # Standard deviation of the normal distribution used for adding noise to random walk (sd < 1)
        self.sd = 0.3
        # Set seed for reproductibility
        self.rng = np.random.default_rng(seed=random_state)

    def move(self):
        # Random angle rotation - Correlated Random Walk
        d_theta = self.rng.normal(0, self.sd) 
        
        # Rotation matrix
        cos_d = np.cos(d_theta)
        sin_d = np.sin(d_theta)
        rotation_matrix = np.array([[cos_d, -sin_d], [sin_d,  cos_d]])
        
        old_pos = self.pos.copy()

        # New acceleration vector and next position
        self.velocity = np.dot(rotation_matrix, self.velocity)
        self.pos = self.pos + self.velocity 

        delta_pos = self.pos - old_pos

        return self.pos.copy(), delta_pos

class Wolf:
    '''
    Used as a memory buffor for 5 last rabbit movements and transforms the output data into model readable shape.
    '''
    
    def __init__(self):
        self.history = []
        
    def update_history(self, delta):
        # Store 5 last Rabbit movements
        self.history.append(delta)
        if len(self.history) > 5:
            # Space for next positions
            self.history.pop(0)     
    
    # Parses position data into scikitlearn friendly format 
    # [(x1,y1),(x2,y2),...] - > [x1;y1;x2;y2;...]
    def prepare_input_data(self):
        if len(self.history) < 5:
            return None
        return np.array(self.history).flatten().reshape(1, -1)

# --- Path generation test ---
if __name__ == "__main__":
    rabbit = Rabbit(start_pos=(0, 0), speed=0.5, random_state=42)
    x_coords, y_coords = [rabbit.pos[0]], [rabbit.pos[1]]

    for _ in range(50):
        pos, delta = rabbit.move() 
        x_coords.append(pos[0])
        y_coords.append(pos[1])

    plt.figure(figsize=(6, 4))
    plt.plot(x_coords, y_coords, 'g-o', markersize=3)
    plt.title("Test płynnego ruchu")
    plt.axis('equal')
    plt.show()