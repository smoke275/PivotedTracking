#!/usr/bin/env python3
"""
Simulation Configuration Helper
Contains all constants, colors, and configuration variables for the simple agent simulation.
This helps keep the main simulation file clean and organized.
"""

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

# Screen dimensions
SIDEBAR_WIDTH = 250          # Width of the information sidebar
ENVIRONMENT_WIDTH = 1280     # Width of the environment area  
ENVIRONMENT_HEIGHT = 720     # Height of the environment area
WINDOW_WIDTH = ENVIRONMENT_WIDTH + SIDEBAR_WIDTH  # Total window width including sidebar
WINDOW_HEIGHT = ENVIRONMENT_HEIGHT               # Window height

# Font configuration
FONT_NAME = 'Arial'
FONT_SIZE = 12

# Frame rate
TARGET_FPS = 60

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Agent visual properties
AGENT_RADIUS = 16
AGENT_COLOR = (255, 0, 255)     # Magenta for agent 1
AGENT2_COLOR = (0, 255, 255)    # Cyan for agent 2

# Agent file persistence
AGENT_STATE_FILE = 'agent_state.pkl'
AGENT2_STATE_FILE = 'agent2_state.pkl'

# =============================================================================
# VISUALIZATION THRESHOLDS
# =============================================================================

# Node visualization thresholds
DISTANCE_THRESHOLD = 50.0   # Distance threshold in pixels - nodes closer than this are more transparent
TIME_THRESHOLD = 100.0      # Time threshold in seconds - nodes taking longer than this are marked invalid and more transparent

# =============================================================================
# TIMING CONFIGURATION
# =============================================================================

# Update intervals (in seconds)
POSITION_UPDATE_INTERVAL = 0.5      # Update position evaluator every 0.5 seconds
TRAVEL_TIME_UPDATE_INTERVAL = 2.0   # Update travel times every 2 seconds
VISIBILITY_UPDATE_INTERVAL = 0.1    # Update visibility every 0.1 seconds

# =============================================================================
# RRT* CONFIGURATION
# =============================================================================

# RRT* tree parameters
RRT_MAX_NODES = 200
RRT_STEP_SIZE = 10.0
RRT_SEARCH_RADIUS = 25.0
RRT_FORWARD_BIAS = 0.7              # 70% of samples biased forward
RRT_FORWARD_CONE_ANGLE = 1.5708     # π/2 radians = 90 degrees

# =============================================================================
# TRAJECTORY OPTIMIZATION CONFIGURATION
# =============================================================================

# Trajectory constraints
MAX_ACCELERATION = 50.0             # pixels/second²
TRAJECTORY_NUM_POINTS = 150         # Number of points in optimized trajectory

# =============================================================================
# CLOSEST NODE CACHE CONFIGURATION
# =============================================================================

# Cache thresholds
CLOSEST_NODE_MOVEMENT_THRESHOLD = 12.0  # Recalculate when agent moves 12 pixels
CLOSEST_NODE_TIME_THRESHOLD = 1.5       # Recalculate after 1.5 seconds max

# =============================================================================
# VISIBILITY SYSTEM CONFIGURATION
# =============================================================================

# Visibility parameters
DEFAULT_VISIBILITY_RANGE = 200.0    # Maximum visibility distance
VISIBILITY_NUM_RAYS = 100           # Number of rays for 360° visibility (3.6° increments)
MIN_GAP_DISTANCE = 30               # Minimum distance difference to consider a visibility gap

# Visibility RRT parameters - optimized for unknown area filling
VISIBILITY_RRT_MAX_NODES = 400      # More nodes for better area coverage
VISIBILITY_RRT_STEP_SIZE = 10.0     # Smaller step size for better resolution
VISIBILITY_RRT_SEARCH_RADIUS = 30.0 # Larger search radius for better connectivity
VISIBILITY_RRT_BIAS_STRENGTH = 0.7  # 70% bias toward unknown areas (area filling)
VISIBILITY_RRT_UNKNOWN_AREA_MIN = 0.6  # Start sampling from 60% of visibility range (unknown areas)

# =============================================================================
# REACHABILITY SYSTEM CONFIGURATION
# =============================================================================

# Reachability mask configuration
REACHABILITY_MASK_NAME = "unicycle_grid"
REACHABILITY_OVERLAY_ALPHA = 180    # Transparency for reachability grid overlay
REACHABILITY_MAX_DISPLAY_SIZE = 300 # Maximum pixel size for the overlay
REACHABILITY_BORDER_COLOR = (255, 255, 255, 150)  # White border with transparency
REACHABILITY_BORDER_WIDTH = 2       # Border line width

# =============================================================================
# INFO PANEL CONFIGURATION
# =============================================================================

# Scrollable info panel
INFO_PANEL_LINE_SPACING = 15        # Pixels between lines
INFO_PANEL_SCROLL_SPEED = 3         # Lines to scroll per wheel event
INFO_PANEL_MARGIN = 10              # Margin around text
INFO_PANEL_BACKGROUND_ALPHA = 180   # Transparency of info panel background

# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Basic visibility colors
VISIBILITY_COLORS = {
    'ray': (0, 255, 255, 100),           # Cyan, semi-transparent
    'blocked_ray': (255, 100, 100, 150), # Red, more visible
    'visibility_area': (0, 255, 255, 30), # Light cyan, very transparent
    'breakoff_point': (255, 255, 0, 200), # Yellow, highly visible for breakoff points
    'breakoff_line': (255, 165, 0, 180)   # Orange for breakoff lines
}

# Breakoff distance concentric circles
BREAKOFF_CIRCLE_COLOR = (255, 255, 255, 80)  # White, semi-transparent
BREAKOFF_CIRCLE_WIDTH = 2                     # Circle line width

# 4-category breakoff point colors based on orientation and distance transition
BREAKOFF_CATEGORY_COLORS = {
    # Clockwise transitions
    'clockwise_near_far': {
        'point': (255, 50, 50),           # Bright red for clockwise near-to-far
        'line': (255, 100, 100, 180),     # Light red for lines
        'middle': (200, 40, 40)           # Darker red for middle circle
    },
    'clockwise_far_near': {
        'point': (255, 150, 50),          # Orange-red for clockwise far-to-near  
        'line': (255, 180, 100, 180),     # Light orange-red for lines
        'middle': (200, 120, 40)          # Darker orange-red for middle circle
    },
    # Counterclockwise transitions
    'counterclockwise_near_far': {
        'point': (50, 255, 50),           # Bright green for counterclockwise near-to-far
        'line': (100, 255, 100, 180),     # Light green for lines
        'middle': (40, 200, 40)           # Darker green for middle circle
    },
    'counterclockwise_far_near': {
        'point': (50, 255, 150),          # Blue-green for counterclockwise far-to-near
        'line': (100, 255, 180, 180),     # Light blue-green for lines
        'middle': (40, 200, 120)          # Darker blue-green for middle circle
    },
    # Fallback colors
    'unknown_near_far': {
        'point': (255, 255, 0),           # Yellow fallback
        'line': (255, 255, 100, 180),     # Light yellow for lines
        'middle': (200, 200, 0)           # Darker yellow for middle circle
    },
    'unknown_far_near': {
        'point': (255, 200, 0),           # Orange-yellow fallback
        'line': (255, 220, 100, 180),     # Light orange-yellow for lines
        'middle': (200, 160, 0)           # Darker orange-yellow for middle circle
    }
}

# RRT tree colors
RRT_TREE_COLORS = {
    "agent1": (100, 255, 100),  # Light green for agent 1
    "agent2": (100, 100, 255),  # Light blue for agent 2
}

# Strategic analysis highlighting colors
STRATEGIC_COLORS = {
    'worst_node': (255, 50, 50),        # Red highlight for worst node
    'worst_10_nodes': (255, 140, 0),    # Orange highlight for worst 10 nodes
    'inner_circle': (255, 255, 255),    # White inner circle
}

# Trajectory visualization
TRAJECTORY_COLOR = (255, 0, 255)       # Magenta for optimized trajectory
TRAJECTORY_LINE_WIDTH = 3

# Node transparency levels
NODE_TRANSPARENCY = {
    'full_opacity': 255,
    'close_nodes': 120,          # More transparent for close nodes
    'invalid_nodes': 60,         # Most transparent for invalid (slow) nodes
    'edge_close': 100,           # Edge transparency for close nodes  
    'edge_invalid': 80           # Edge transparency for invalid nodes
}

# =============================================================================
# DEFAULT INITIAL STATES
# =============================================================================

# Default visibility system state
DEFAULT_VISIBILITY_STATE = {
    'show_evader_visibility': False,
    'show_visibility_rays': True,
    'show_visibility_area': True,
    'visibility_range': DEFAULT_VISIBILITY_RANGE
}

# Default display toggles
DEFAULT_DISPLAY_STATE = {
    'show_info': True,
    'show_map_graph': False,
    'show_rrt_trees': True,
    'show_trajectory': True,
    'show_reachability_mask': False
}

# Default agent positions
DEFAULT_AGENT_POSITIONS = {
    'agent1': (100, 100),
    'agent2': (200, 200)
}

# =============================================================================
# CONTROL HELP TEXT
# =============================================================================

CONTROL_HELP_TEXT = {
    'basic_controls': [
        "Controls:",
        "Arrow Keys: Control Agent 1 (Magenta)",
        "WASD Keys: Control Agent 2 (Cyan)",
        "G: Toggle map graph display",
        "R: Toggle RRT* trees display",
        "T: Regenerate RRT* trees + auto-map to graph",
        "U: Force update travel times + auto-map to graph",
        "P: Strategic analysis (Agent 1 pursues Agent 2)",
        "M: Toggle reachability grid overlay (Agent 2)",
        "V: Toggle evader visibility (Agent 2 360° rays)",
        "B: Generate visibility segment RRT trees",
        "N: Toggle visibility segment RRT trees display",
        "Q: Toggle measurement tool (click & drag to measure)",
        "I: Toggle info display",
        "S: Save agent states to file",
        "L: Save environment lines to file",
        "K: Save visibility polygon to file",
        "Mouse Click: Select RRT node (auto-gen traj)",
        "C: Clear selected path & trajectory",
        "X: Clear closest node cache",
        "Z: Force rebuild spatial index",
        "ESC: Quit simulation"
    ],
    
    'reachability_help': [
        "Reachability Grid Overlay:",
        "  M: Show/hide Agent 2's reachability grid",
        "  Grid follows Agent 2 position and orientation",
        "  Shows probability-based coloring (hot colormap)"
    ],
    
    'visibility_help': [
        "Evader Visibility System:",
        f"  V: Show/hide Agent 2's 360° visibility rays",
        f"  L: Save environment lines to timestamped file",
        f"  K: Save visibility polygon to timestamped file",
        f"  Current range: {DEFAULT_VISIBILITY_RANGE:.0f} pixels",
        f"  Using {VISIBILITY_NUM_RAYS} rays ({360/VISIBILITY_NUM_RAYS:.1f}° increments) for optimal accuracy/performance",
        "  Rays blocked by walls, pass through doors",
        "  Breakoff Point Categories (relative to Agent 2's orientation):",
        "    Red circles/lines = clockwise near-to-far transitions",
        "    Orange-red circles/lines = clockwise far-to-near transitions", 
        "    Green circles/lines = counterclockwise near-to-far transitions",
        "    Blue-green circles/lines = counterclockwise far-to-near transitions",
        "  Updates in real-time as Agent 2 moves and rotates"
    ],
    
    'measurement_help': [
        "Measurement Tool:",
        "  Q: Toggle measurement tool on/off",
        "  Click & drag to measure distances between points",
        "  Yellow line shows measurement with green start, red end markers",
        "  Real-time distance display during drag",
        "  Final distance shown in pixels and estimated world units",
        "  Works only in environment area (not sidebar)"
    ],
    
    'node_visualization_help': [
        "Node Visualization:",
        f"  Distance threshold: {DISTANCE_THRESHOLD:.1f}px (nodes closer = more transparent)",
        f"  Time threshold: {TIME_THRESHOLD:.1f}s (nodes slower = invalid/transparent)",
        "  Invalid nodes are marked with 'X' and high transparency",
        "  Strategic highlighting: Worst node = red, Worst 10 nodes = orange",
        f"  Strategic analysis only considers nodes ≤{TIME_THRESHOLD:.1f}s (valid nodes)",
        "Spatial Index System:",
        "  🎯 O(1) real-time agent position tracking (silent for performance)", 
        "  ⚡ Grid-based spatial hashing for fast lookups in position evaluator",
        "  📍 Check info panel (I key) to see spatial index statistics"
    ],
    
    'overlay_api_help': [
        "Reachability Overlay API:",
        "  🎯 Automatically initialized at startup for path analysis",
        "  📐 Config: 32px clip, 120x120→400x400 resize, max_pool downsampling",
        "  ⚖️ Probability weighting: (0.55, 0.8) for risk-aware analysis",
        "  🔧 Use get_overlay_api() to access the configured instance",
        "  📊 Call overlay_api.process_paths_with_reachability() for processing",
        "  ⚡ Ready for multiple path processing calls without reconfiguration"
    ]
}

# =============================================================================
# PERFORMANCE OPTIMIZATIONS
# =============================================================================

# Batch processing sizes
BATCH_SIZE_RECTS = 100              # Draw rects in batches for better performance
BATCH_SIZE_LINES = 50               # Draw lines in batches

# Cache sizes
MAX_VISIBILITY_CACHE_SIZE = 1000    # Maximum cached visibility calculations
MAX_TRAJECTORY_CACHE_SIZE = 100     # Maximum cached trajectory calculations

# =============================================================================
# DEBUGGING AND LOGGING
# =============================================================================

# Debug flags
DEBUG_VISIBILITY = False
DEBUG_TRAJECTORY = False  
DEBUG_RRT = False
DEBUG_PERFORMANCE = False

# Log levels
LOG_LEVEL_INFO = 1
LOG_LEVEL_DEBUG = 2
LOG_LEVEL_VERBOSE = 3
DEFAULT_LOG_LEVEL = LOG_LEVEL_INFO
