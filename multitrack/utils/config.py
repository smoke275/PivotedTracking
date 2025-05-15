"""
Constants and configuration settings for the unicycle simulation and controllers.
This file centralizes all configurable parameters to make adjustments easier.
"""
import math
import pygame

# Screen dimensions
WIDTH = 1280  # Updated to match the simulation window size
HEIGHT = 720  # Updated to match the simulation window size
SCREEN_RECT = pygame.Rect(0, 0, WIDTH, HEIGHT)

# FPS and timing
TARGET_FPS = 45                # Target frames per second

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)  # Color for the follower agent

# Environment colors
WALL_COLOR = (120, 120, 120)    # Gray walls
FLOOR_COLOR = (240, 230, 210)   # Beige floor
DOOR_COLOR = (210, 180, 140)    # Tan door
WINDOW_COLOR = (173, 216, 230)  # Light blue windows

# Leader agent settings
LEADER_LINEAR_VEL = 50.0        # Maximum linear velocity for leader
LEADER_ANGULAR_VEL = 1.0        # Maximum angular velocity for leader

# Follower agent settings
FOLLOWER_ENABLED = True                  # Enable/disable follower agent
FOLLOWER_TARGET_DISTANCE = 50.0         # Default following distance
FOLLOWER_LINEAR_VEL_MIN = 0.0            # Minimum linear velocity (changed from -50.0 to enforce forward-only motion)
FOLLOWER_LINEAR_VEL_MAX = 50.0           # Maximum linear velocity
FOLLOWER_ANGULAR_VEL_MIN = -1.2          # Minimum angular velocity (increased for sharper turns)
FOLLOWER_ANGULAR_VEL_MAX = 1.2           # Maximum angular velocity (increased for sharper turns)
FOLLOWER_LINEAR_NOISE_SIGMA = 10.0       # Noise std dev for linear velocity in MPPI
FOLLOWER_ANGULAR_NOISE_SIGMA = 0.5       # Increased noise for more exploration of turning options
FOLLOWER_SAFETY_DISTANCE = 40.0          # Increased minimum safety distance for collision avoidance
FOLLOWER_STOPPING_RADIUS = 70.0          # Radius around stationary visitor to stop moving
FOLLOWER_MIN_DISTANCE = 50.0             # Minimum allowed following distance
FOLLOWER_MAX_DISTANCE = 200.0            # Maximum allowed following distance
FOLLOWER_SEARCH_DURATION = 100           # Frames to continue searching after losing sight of target
FOLLOWER_PROXIMITY_PENALTY = 20.0        # Heavy penalty factor for getting too close to visitor

# MPPI controller settings
MPPI_HORIZON = 35                       # Reduced from 30 for better performance
MPPI_SAMPLES = 15000                    # Reduced from 1000 for better performance
MPPI_LAMBDA = 0.05                      # Temperature for softmax weighting - decreased for smoother control
MPPI_WEIGHT_POSITION = 1.0              # Weight for position tracking
MPPI_WEIGHT_HEADING = 0.5               # Weight for heading alignment
MPPI_WEIGHT_CONTROL = 0.1               # Weight for control effort - increased for smoother control
MPPI_WEIGHT_COLLISION = 10.0            # Weight for collision avoidance
MPPI_WEIGHT_FORWARD = 0.3               # Weight for forward direction incentive
MPPI_USE_GPU = True                     # Enable GPU acceleration
MPPI_GPU_BATCH_SIZE = 12288             # Process samples in batches for better GPU memory management (optimized for RTX A3000)
MPPI_USE_ASYNC = True                   # Use asynchronous computation where possible
MPPI_CACHE_SIZE = 5                     # Cache recent computations for reuse
MPPI_MULTITHREAD_ENABLED = True         # Enable multithreading for CPU operations
MPPI_THREAD_POOL_SIZE = 8               # Number of threads to use (None = auto-detect)
MPPI_THREAD_CHUNK_SIZE = 1000           # Chunk size for dividing work among threads

# Kalman filter settings
KF_MEASUREMENT_INTERVAL = 0.5           # How often to take measurements (seconds)
KF_MEASUREMENT_NOISE_POS = 2.0          # Position measurement noise
KF_MEASUREMENT_NOISE_ANGLE = 0.1        # Angle measurement noise
KF_PROCESS_NOISE_POS = 2.0              # Process noise for position
KF_PROCESS_NOISE_ANGLE = 0.1            # Process noise for angle

# Measurement settings
DEFAULT_MEASUREMENT_INTERVAL = 0.2      # Default measurement interval in seconds
MIN_MEASUREMENT_INTERVAL = 0.1          # Minimum measurement interval (fastest rate)
MAX_MEASUREMENT_INTERVAL = 2.0          # Maximum measurement interval (slowest rate)

# Visualization settings
SHOW_PREDICTIONS = True                 # Show Kalman filter predictions
PREDICTION_STEPS = 20                   # Number of steps to predict into future
SHOW_UNCERTAINTY = True                 # Show uncertainty ellipse
SHOW_MPPI_PREDICTIONS = True            # Show MPPI predictions
PREDICTION_COLOR = CYAN                 # Color for Kalman predictions
UNCERTAINTY_COLOR = (100, 100, 255, 100)  # Color for uncertainty ellipse (with alpha)
MPPI_PREDICTION_COLOR = (255, 100, 100)   # Color for MPPI predictions

# Vision parameters
DEFAULT_VISION_RANGE = 800      # Initial vision range in pixels
VISION_ANGLE = math.pi/2        # 60 degrees field of view

# Secondary Camera parameters
SECONDARY_CAMERA_MAX_ANGULAR_VEL = 4.0  # Maximum angular velocity (radians/sec)
SECONDARY_CAMERA_ANGULAR_ACCEL = 1.0   # Angular acceleration (radians/sec²)
SECONDARY_CAMERA_ANGULAR_DECEL = 0.5    # Angular deceleration (radians/sec²)

# Camera auto-tracking parameters
CAMERA_AUTO_TRACK_ENABLED = True  # Default auto-tracking state
CAMERA_PID_P = 3.0  # Proportional gain for camera tracking
CAMERA_PID_I = 0.1  # Integral gain for camera tracking
CAMERA_PID_D = 0.5  # Derivative gain for camera tracking
CAMERA_SEARCH_SPEED = 1.0  # Angular velocity for camera search behavior (radians/sec)
CAMERA_MAX_ERROR_INTEGRAL = 1.0  # Maximum error integral to prevent windup
CAMERA_TRACK_TIMEOUT = 60  # Frames to continue tracking after losing sight of visitor

# Agent parameters
AGENT_SIZE = 20                 # Size of the agent
MAX_VELOCITY = 5                # Maximum speed
MAX_STEERING_ANGLE = 0.15       # Maximum steering angle (radians per frame)
TURNING_RADIUS = 25             # Minimum turning radius in pixels
ACCELERATION = 0.5              # Acceleration rate
DECELERATION = 0.3              # Deceleration rate
STEERING_SPEED = 0.03           # How quickly steering angle changes

# Vision cone rendering
NUM_VISION_RAYS = 40            # Number of rays for vision cone
VISION_TRANSPARENCY = 40        # Alpha value for vision cone (0-255)
VISION_MULTITHREAD_ENABLED = True  # Enable multithreading for vision cone generation
VISION_THREAD_POOL_SIZE = 4     # Number of threads to use for vision cone (None = auto-detect)
VISION_THREAD_CHUNK_SIZE = 8    # Number of rays per thread chunk

# Map Graph parameters
MAP_GRAPH_GRID_SIZE = 120        # Resolution of the sampling grid (higher values = denser grid)
MAP_GRAPH_MAX_EDGE_DISTANCE = 80  # Maximum distance for connecting nodes (lower = more precise paths)
MAP_GRAPH_MAX_CONNECTIONS = 12    # Maximum connections per node (higher = denser graph)
MAP_GRAPH_NODE_COLOR = (100, 100, 255)  # Color for graph nodes
MAP_GRAPH_EDGE_COLOR = (50, 50, 200)    # Color for graph edges
MAP_GRAPH_CACHE_ENABLED = True  # Whether to cache map graphs between runs
MAP_GRAPH_CACHE_FILE = "map_graph_cache.pkl"  # File to store cached map graph
MAP_GRAPH_INSPECTION_CACHE_FILE = "inspection_map_graph_cache.pkl"  # File for inspection simulation
MAP_GRAPH_VISIBILITY_CACHE_FILE = "graph_visibility_cache.pkl"  # File to store node visibility data
MAP_GRAPH_VISIBILITY_RANGE = 1600  # Maximum visibility range in pixels
MAP_GRAPH_MULTICORE_DEFAULT = True  # Use multicore processing by default

# Particle effects
PARTICLE_LIFETIME = 10          # How long particles last
PARTICLE_VELOCITY_FACTOR = 0.2  # Speed of particles relative to agent

# Font settings
FONT_SIZE = 24