import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import entropy

def compute_expression(P, g):
    H_P = entropy(P, base=2)

    # Exclude 0th entry and normalize
    P_minus_0 = P[1:]
    P_minus_0 /= P_minus_0.sum()

    # Uniform over 9 elements
    U = np.ones_like(P_minus_0) / len(P_minus_0)

    D_KL = entropy(P_minus_0, U, base=2)
    return H_P  + g * D_KL

# Initial values
init_probs = np.ones(10) / 10
g = 2.0

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 4))
plt.subplots_adjust(left=0.1, bottom=0.4)

text = ax.text(0.5, 0.5, '', ha='center', va='center', fontsize=14)
ax.axis('off')

# --- Slider axes ---
sliders = []
slider_axes = []

for i in range(10):
    ax_slider = plt.axes([0.1, 0.35 - i * 0.025, 0.8, 0.02])
    slider = Slider(ax_slider, f'P[{i}]', 0.001, 1.0, valinit=init_probs[i])
    sliders.append(slider)
    slider_axes.append(ax_slider)

# --- g slider ---
g_ax = plt.axes([0.1, 0.1, 0.8, 0.03])
g_slider = Slider(g_ax, 'g', 0.0, 5.0, valinit=g)

def update(val):
    # Read values from sliders
    probs = np.array([s.val for s in sliders])
    probs /= probs.sum()  # Normalize to sum to 1

    g_val = g_slider.val
    result = compute_expression(probs, g_val)

    # Update the text
    text.set_text(f"H(P) + g·DKL(P_-0 || U)\n= {result:.4f}")

# Attach update callback
for s in sliders:
    s.on_changed(update)
g_slider.on_changed(update)

# Initial draw
update(None)
plt.show()
