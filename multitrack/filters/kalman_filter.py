"""
Kalman filter implementation for unicycle model state estimation and prediction.
This file contains a Kalman filter class designed to work with the unicycle model,
tracking the state [x, y, theta, v, omega] and predicting future states.
"""
import numpy as np
from math import sin, cos, pi

class UnicycleKalmanFilter:
    def __init__(self, initial_state, dt=0.1):
        """
        Initialize Kalman filter for unicycle model
        
        Parameters:
        - initial_state: numpy array [x, y, theta, v]
        - dt: time step (default 0.1)
        """
        # State: [x, y, theta, v, omega]
        # Adding omega to the state vector for better prediction
        self.state = np.zeros(5)
        self.state[:4] = initial_state
        self.state[4] = 0.0  # Initial angular velocity
        
        self.dt = dt
        
        # Initialize state covariance matrix
        self.P = np.diag([10.0, 10.0, 0.2, 5.0, 0.2])  # Initial uncertainty
        
        # Process noise covariance matrix
        self.Q = np.diag([2.0, 2.0, 0.1, 5.0, 0.1])  # Adjust based on expected noise
        
        # Measurement noise covariance matrix
        # For position and heading measurements (increased for bad sensor)
        self.R = np.diag([25.0, 25.0, 0.5])  # Increased from [5.0, 5.0, 0.1]
        
        # Control input matrix - not needed for our implementation
        # as we incorporate controls directly in the process model
        
        # History for analysis
        self.history = {
            'states': [],
            'predictions': [],
            'measurements': [],
            'covariances': []
        }
    
    def predict(self, control_input=None, num_steps=1):
        """
        Predict future states using the current state and control input
        
        Parameters:
        - control_input: numpy array [v, omega] or None
        - num_steps: number of steps to predict ahead
        
        Returns:
        - Array of predicted states (n+1 states including current)
        """
        predictions = []
        current_state = self.state.copy()
        predictions.append(current_state.copy())
        
        # If no control input is provided, use the current velocities
        if control_input is None:
            v = current_state[3]
            omega = current_state[4]
        else:
            v, omega = control_input
        
        # Generate predictions for each time step
        for _ in range(num_steps):
            # Non-linear unicycle model
            theta = current_state[2]
            
            # State transition
            next_state = current_state.copy()
            next_state[0] += v * cos(theta) * self.dt
            next_state[1] += v * sin(theta) * self.dt
            next_state[2] += omega * self.dt
            next_state[2] = (next_state[2] + pi) % (2 * pi) - pi  # Normalize angle
            next_state[3] = v  # Linear velocity
            next_state[4] = omega  # Angular velocity
            
            current_state = next_state
            predictions.append(current_state.copy())
        
        return predictions
    
    def jacobian(self, state, control_input=None):
        """
        Compute the Jacobian of the process model at the given state
        This is used for the Extended Kalman Filter update
        
        Parameters:
        - state: numpy array [x, y, theta, v, omega]
        - control_input: numpy array [v, omega] or None
        
        Returns:
        - F: Jacobian matrix (5x5)
        """
        # If no control input, use current velocities
        if control_input is None:
            v = state[3]
            omega = state[4]
        else:
            v, omega = control_input
            
        theta = state[2]
        
        # Initialize Jacobian with identity matrix
        F = np.eye(5)
        
        # Partial derivatives
        F[0, 2] = -v * sin(theta) * self.dt
        F[0, 3] = cos(theta) * self.dt
        F[1, 2] = v * cos(theta) * self.dt
        F[1, 3] = sin(theta) * self.dt
        F[2, 4] = self.dt
        
        return F
    
    def update(self, measurement, control_input=None):
        """
        Update the Kalman filter with a new measurement
        
        Parameters:
        - measurement: numpy array [x, y, theta] or None
        - control_input: numpy array [v, omega] or None
        
        Returns:
        - The updated state estimate
        """
        # Save current state for history
        self.history['states'].append(self.state.copy())
        
        # Prediction step
        self.predict_step(control_input)
        
        # Update step (if measurement is provided)
        if measurement is not None:
            self.update_step(measurement)
        
        # Save covariance for history
        self.history['covariances'].append(np.diag(self.P).copy())
        
        return self.state.copy()
    
    def predict_step(self, control_input=None):
        """
        Perform the prediction step of the Extended Kalman Filter
        
        Parameters:
        - control_input: numpy array [v, omega] or None
        """
        # If no control input is provided, use the current velocities
        if control_input is None:
            v = self.state[3]
            omega = self.state[4]
        else:
            v, omega = control_input
            
        # Get current state
        theta = self.state[2]
        
        # Predict next state using unicycle model
        self.state[0] += v * cos(theta) * self.dt
        self.state[1] += v * sin(theta) * self.dt
        self.state[2] += omega * self.dt
        self.state[2] = (self.state[2] + pi) % (2 * pi) - pi  # Normalize angle
        self.state[3] = v  # Update velocity
        self.state[4] = omega  # Update angular velocity
        
        # Compute Jacobian
        F = self.jacobian(self.state, control_input)
        
        # Update covariance matrix
        self.P = F @ self.P @ F.T + self.Q
        
        # Save prediction for history
        self.history['predictions'].append(self.state.copy())
    
    def update_step(self, measurement):
        """
        Perform the update step of the Extended Kalman Filter
        
        Parameters:
        - measurement: numpy array [x, y, theta]
        """
        # Save measurement for history
        self.history['measurements'].append(np.append(measurement, [0, 0]))
        
        # Measurement matrix - we're only measuring position and heading
        H = np.zeros((3, 5))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # theta
        
        # Calculate measurement residual
        z = measurement
        h = np.array([self.state[0], self.state[1], self.state[2]])
        y = z - h
        
        # Normalize angle difference to [-pi, pi]
        y[2] = (y[2] + pi) % (2 * pi) - pi
        
        # Calculate Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state += K @ y
        
        # Normalize angle again after update
        self.state[2] = (self.state[2] + pi) % (2 * pi) - pi
        
        # Update covariance matrix
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
    
    def get_prediction_ellipse(self, confidence=0.95):
        """
        Get the covariance ellipse for position uncertainty
        
        Parameters:
        - confidence: Confidence level (default 0.95 for 95%)
        
        Returns:
        - points: List of points defining the ellipse
        """
        from scipy.stats import chi2
        
        # For 2D position, a 95% confidence ellipse needs chi2(2) = 5.991
        s = chi2.ppf(confidence, 2)
        
        # Get position covariance submatrix
        pos_cov = self.P[:2, :2]
        
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eig(pos_cov)
        
        # Ensure eigenvalues are positive (they should be for covariance matrices)
        eigvals = np.maximum(eigvals, 0)
        
        # Calculate ellipse parameters
        a = np.sqrt(s * eigvals[0])  # Semi-major axis
        b = np.sqrt(s * eigvals[1])  # Semi-minor axis
        
        # Get rotation angle from first eigenvector
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        
        # Generate ellipse points
        t = np.linspace(0, 2*pi, 100)
        ellipse_x = a * np.cos(t)
        ellipse_y = b * np.sin(t)
        
        # Rotate ellipse
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        points = np.vstack([ellipse_x, ellipse_y])
        points = R @ points
        
        # Translate to state position
        points[0, :] += self.state[0]
        points[1, :] += self.state[1]
        
        return points.T  # Return as Nx2 array for easier plotting

    def get_state_with_confidence(self):
        """
        Returns the current state estimate with confidence intervals
        
        Returns:
        - state: Current state estimate
        - uncertainty: Standard deviations for each state variable
        """
        state = self.state.copy()
        uncertainty = np.sqrt(np.diag(self.P))
        
        return state, uncertainty
        
    def calculate_entropy(self, position_only=True):
        """
        Calculate the differential entropy of the state distribution
        
        For a multivariate Gaussian, the differential entropy is:
        H(X) = 0.5 * log((2*pi*e)^n * det(P))
        
        Parameters:
        - position_only: If True, calculate entropy only for x,y position (default)
                         If False, calculate entropy for the full state
        
        Returns:
        - entropy: The differential entropy in nats (natural units)
        """
        if position_only:
            # Extract just the position covariance (2x2 submatrix)
            cov = self.P[:2, :2]
            n = 2
        else:
            # Use the full covariance
            cov = self.P
            n = cov.shape[0]
        
        # Calculate determinant (handle numerical issues with small values)
        det_cov = max(np.linalg.det(cov), 1e-10)
        
        # Differential entropy formula for multivariate Gaussian
        entropy = 0.5 * np.log((2 * np.pi * np.e)**n * det_cov)
        
        return entropy

    def reset_history(self):
        """Reset the filter history"""
        self.history = {
            'states': [],
            'predictions': [],
            'measurements': [],
            'covariances': []
        }