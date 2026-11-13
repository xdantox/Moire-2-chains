import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Global configuration for bold, publication-quality fonts
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.family': 'sans-serif'
})

fig, ax = plt.subplots(figsize=(22, 18), facecolor='white')
ax.axis('off')

# Improved spring drawing function for symbolic energy channels
def draw_spring(ax, x, y, length, orientation='v', color='#1f77b4', lw=4, state='relaxed'):
    num_coils = 6 if state == 'compressed' else 4
    actual_length = length * 0.6 if state == 'compressed' else length
    
    if orientation == 'v':
        ys = np.linspace(y, y - actual_length, num_coils * 2 + 1)
        xs = np.zeros_like(ys) + x
        xs[1::2] = x + 0.15
        xs[2:-1:2] = x - 0.15
        ax.plot(xs, ys, color=color, lw=lw)
        # End caps
        ax.plot([x - 0.3, x + 0.3], [ys[0], ys[0]], color=color, lw=lw)
        ax.plot([x - 0.3, x + 0.3], [ys[-1], ys[-1]], color=color, lw=lw)
    else:
        xs = np.linspace(x, x + actual_length, num_coils * 2 + 1)
        ys = np.zeros_like(xs) + y
        ys[1::2] = y + 0.15
        ys[2:-1:2] = y - 0.15
        ax.plot(xs, ys, color=color, lw=lw)
        # End caps
        ax.plot([xs[0], xs[0]], [y - 0.3, y + 0.3], color=color, lw=lw)
        ax.plot([xs[-1], xs[-1]], [y - 0.3, y + 0.3], color=color, lw=lw)

# Function to draw the physical magnetic phases
def draw_phase(ax, x_center, y_center, d_gap, epsilon, title, phase_label):
    n_spins = 7
    x_len = 3.5
    a = x_len / (n_spins - 1)
    
    # xs_top is centered. xs_bot is shifted by epsilon.
    xs_top = np.linspace(x_center - x_len/2, x_center + x_len/2, n_spins)
    xs_bot = np.linspace(x_center - x_len/2 - epsilon, x_center + x_len/2 - epsilon, n_spins)
    
    y_top = y_center + d_gap/2
    y_bot = y_center - d_gap/2

    # Draw exchange connections (stronger when closer)
    ex_color = '#2ca02c'
    ex_lw = 4.0 if d_gap < 1.0 else 1.5
    ex_alpha = 0.8 if d_gap < 1.0 else 0.4
    
    for i in range(n_spins):
        ax.plot([xs_top[i], xs_bot[i]], [y_top, y_bot], '--', color=ex_color, alpha=ex_alpha, lw=ex_lw, zorder=1)

    # Chain lines
    ax.plot([xs_top[0] - 0.3, xs_top[-1] + 0.3], [y_top, y_top], color='#1f77b4', lw=4, zorder=2)
    ax.plot([xs_bot[0] - 0.3, xs_bot[-1] + 0.3], [y_bot, y_bot], color='#ff7f0e', lw=4, zorder=2)

    # Spins
    for i in range(n_spins):
        # Top spins (Blue, alternating)
        dy_top = 0.3 if i % 2 == 0 else -0.3
        ax.arrow(xs_top[i], y_top, 0, dy_top, head_width=0.1, head_length=0.15, fc='#1f77b4', ec='#1f77b4', lw=3, zorder=3, length_includes_head=True)
        ax.plot(xs_top[i], y_top, 'o', color='white', mec='#1f77b4', mew=2.5, markersize=8, zorder=4)
        
        # Bottom spins (Orange, alternating opposite)
        dy_bot = -0.3 if i % 2 == 0 else 0.3
        ax.arrow(xs_bot[i], y_bot, 0, dy_bot, head_width=0.1, head_length=0.15, fc='#ff7f0e', ec='#ff7f0e', lw=3, zorder=3, length_includes_head=True)
        ax.plot(xs_bot[i], y_bot, 'o', color='white', mec='#ff7f0e', mew=2.5, markersize=8, zorder=4)

    # Perpendicular Distance d marker
    left_x = min(xs_top[0], xs_bot[0]) - 0.6
    ax.annotate('', xy=(left_x, y_top), xytext=(left_x, y_bot),
                arrowprops=dict(arrowstyle='<|-|>', color='#343a40', lw=2))
    d_label = '$d_1$' if d_gap > 1.0 else '$d_2$'
    ax.text(left_x - 0.2, y_center, d_label, va='center', ha='right', fontsize=24, fontweight='bold', color='#343a40')

    # Longitudinal epsilon marker (showing relative shift)
    # Draw vertical reference lines from the last spin of the top chain
    ref_x = xs_top[-1]
    bot_x = xs_bot[-1]
    
    ax.plot([ref_x, ref_x], [y_top, y_bot - 0.6], ':', color='gray', lw=2)
    ax.plot([bot_x, bot_x], [y_bot, y_bot - 0.6], ':', color='gray', lw=2)
    
    if abs(ref_x - bot_x) > 0.05:
        ax.annotate('', xy=(ref_x, y_bot - 0.4), xytext=(bot_x, y_bot - 0.4),
                    arrowprops=dict(arrowstyle='<|-|>', color='#ff7f0e', lw=2))
        e_label = r'$\epsilon_2$' if abs(epsilon) > 0.3 else r'$\epsilon_1$'
        ax.text((ref_x + bot_x)/2, y_bot - 0.5, e_label, ha='center', va='top', color='#ff7f0e', fontsize=22, fontweight='bold')
    else:
        ax.text(ref_x, y_bot - 0.5, r'$\epsilon_1$', ha='center', va='top', color='#ff7f0e', fontsize=22, fontweight='bold')

    # Phase Titles
    ax.text(x_center, y_center + d_gap/2 + 1.2, phase_label, ha='center', va='bottom', fontsize=24, fontweight='bold')
    ax.text(x_center, y_center + d_gap/2 + 0.6, title, ha='center', va='bottom', fontsize=24)


# Coordinate Layout
x_left, x_right = 5, 17
y_top, y_bot = 14, 4
d_large, d_small = 2.4, 1.0
e1, e2 = 0.0, 0.6

# Draw the 4 Physical Phases
draw_phase(ax, x_left, y_top, d_large, e1, r'$(d_1, \epsilon_1)$', 'PHASE 1')
draw_phase(ax, x_left, y_bot, d_small, e1, r'$(d_2, \epsilon_1)$', 'PHASE 2')
draw_phase(ax, x_right, y_bot, d_small, e2, r'$(d_2, \epsilon_2)$', 'PHASE 3')
draw_phase(ax, x_right, y_top, d_large, e2, r'$(d_1, \epsilon_2)$', 'PHASE 4')

# Central Magnonic Vacuum Box
box_x, box_y = 7.5, 7.5
box_w, box_h = 7.0, 3.0
rect = patches.FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.4", 
                              ec="#adb5bd", fc="#f8f9fa", lw=3, zorder=0)
ax.add_patch(rect)

center_text = (
    "$\\mathbf{Magnonic\\ Vacuum\\ Energy}$\n"
    "$E_{ZP} = E_{ZP}(d, \\epsilon)$\n\n"
    "Central energy reservoir controlled by\n"
    "perpendicular separation ($d$) and\n"
    "longitudinal moiré registry ($\\epsilon$)."
)
ax.text(box_x + box_w/2, box_y + box_h/2, center_text, 
        ha='center', va='center', fontsize=20, linespacing=1.4)

# ==========================================
# TRANSITIONS AND SYMBOLIC ENERGY CHANNELS
# ==========================================
arrow_kwargs = dict(width=3, headwidth=12, headlength=15)

# 1 -> 2 (Left, Decrease d, Perpendicular Work)
ax.annotate('', xy=(x_left-0.5, y_bot+2.0), xytext=(x_left-0.5, y_top-2.0), 
            arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', **arrow_kwargs))
ax.text(x_left-1.0, (y_top+y_bot)/2 + 0.8, "Decrease $d$\n(Fixed $\\epsilon_1$)", ha='right', va='center', fontsize=20, fontweight='bold')
ax.text(x_left-1.0, (y_top+y_bot)/2 - 1.0, "Casimir force\nperpendicular work\n$F_C = -\\frac{\\partial E_{ZP}}{\\partial d}$", ha='right', va='center', fontsize=18, color='#1f77b4')

draw_spring(ax, x_left - 3.5, y_top - 1.5, length=4, orientation='v', color='#1f77b4', state='compressed')
ax.annotate('', xy=(x_left - 3.5, y_bot+2.0), xytext=(x_left - 3.5, y_bot+1.0), arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2))
ax.text(x_left - 3.5, y_top - 0.8, 'Load', color='#1f77b4', ha='center', fontsize=18, fontweight='bold')


# 2 -> 3 (Bottom, Increase epsilon, Longitudinal Work)
ax.annotate('', xy=(x_right-2.5, y_bot-1.0), xytext=(x_left+2.5, y_bot-1.0), 
            arrowprops=dict(facecolor='#ff7f0e', edgecolor='#ff7f0e', **arrow_kwargs))
ax.text((x_left+x_right)/2, y_bot-0.3, "Increase $\\epsilon$\n(Fixed $d_2$)", ha='center', va='bottom', fontsize=20, fontweight='bold')
ax.text((x_left+x_right)/2, y_bot-1.6, "Casimir stress — longitudinal work\n$\\sigma_C = -\\frac{\\partial E_{ZP}}{\\partial \\epsilon}$", ha='center', va='top', fontsize=18, color='#ff7f0e')

draw_spring(ax, (x_left+x_right)/2 - 1.5, y_bot - 3.2, length=4, orientation='h', color='#ff7f0e', state='compressed')
ax.annotate('', xy=((x_left+x_right)/2 - 2.5, y_bot - 3.2), xytext=((x_left+x_right)/2 - 3.5, y_bot - 3.2), arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=2))
ax.text((x_left+x_right)/2 + 1.8, y_bot - 3.2, 'Load', color='#ff7f0e', va='center', ha='left', fontsize=18, fontweight='bold')


# 3 -> 4 (Right, Increase d, Perpendicular Work)
ax.annotate('', xy=(x_right+0.5, y_top-2.0), xytext=(x_right+0.5, y_bot+2.0), 
            arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', **arrow_kwargs))
ax.text(x_right+1.0, (y_top+y_bot)/2 + 0.8, "Increase $d$\n(Fixed $\\epsilon_2$)", ha='left', va='center', fontsize=20, fontweight='bold')
ax.text(x_right+1.0, (y_top+y_bot)/2 - 1.0, "Casimir force\nperpendicular work\n$F_C = -\\frac{\\partial E_{ZP}}{\\partial d}$", ha='left', va='center', fontsize=18, color='#1f77b4')

draw_spring(ax, x_right + 3.5, y_top - 1.5, length=4, orientation='v', color='#1f77b4', state='relaxed')
ax.annotate('', xy=(x_right + 3.5, y_top - 0.5), xytext=(x_right + 3.5, y_top - 1.2), arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2))
ax.text(x_right + 3.5, y_top - 0.2, 'Unload', color='#1f77b4', ha='center', fontsize=18, fontweight='bold')


# 4 -> 1 (Top, Decrease epsilon, Longitudinal Work)
ax.annotate('', xy=(x_left+2.5, y_top+1.2), xytext=(x_right-2.5, y_top+1.2), 
            arrowprops=dict(facecolor='#ff7f0e', edgecolor='#ff7f0e', **arrow_kwargs))
ax.text((x_left+x_right)/2, y_top+1.8, "Decrease $\\epsilon$\n(Fixed $d_1$)", ha='center', va='bottom', fontsize=20, fontweight='bold')
ax.text((x_left+x_right)/2, y_top+0.6, "Casimir stress — longitudinal work\n$\\sigma_C = -\\frac{\\partial E_{ZP}}{\\partial \\epsilon}$", ha='center', va='top', fontsize=18, color='#ff7f0e')

draw_spring(ax, (x_left+x_right)/2 - 2.0, y_top + 3.2, length=4, orientation='h', color='#ff7f0e', state='relaxed')
ax.annotate('', xy=((x_left+x_right)/2 - 3.0, y_top + 3.2), xytext=((x_left+x_right)/2 - 2.2, y_top + 3.2), arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=2))
ax.text((x_left+x_right)/2 + 2.5, y_top + 3.2, 'Unload', color='#ff7f0e', va='center', ha='left', fontsize=18, fontweight='bold')

# Main Title
plt.text(11, 19.5, "Magnonic Casimir Cycle in Moiré Antiferromagnets", ha='center', fontsize=36, fontweight='bold')
plt.text(11, 18.7, "Cyclic energy exchange via perpendicular force and longitudinal stress", ha='center', fontsize=24, color='#495057')

plt.xlim(0, 22)
plt.ylim(0, 20)
plt.tight_layout()

# Save highly legible outputs
plt.savefig('magnonic_casimir_cycle_final.svg', format='svg', bbox_inches='tight')
plt.savefig('magnonic_casimir_cycle_final.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig('magnonic_casimir_cycle_final.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)
print("Files generated.")