import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from io import BytesIO
import tempfile
import os

SKELETON_EDGES = [
    # Spine
    (0, 1), (1, 20), (20, 2), (2, 3),
    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),
    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19),
]

def create_animation(skeleton_seq: np.ndarray, action_name: str, fps: int = 15) -> str:
    T = skeleton_seq.shape[0]

    fig = plt.figure(figsize=(6, 6), facecolor='#0d0d0d')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d0d0d')

    # Style
    ax.tick_params(colors='#555555')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1a1a1a')
    ax.yaxis.pane.set_edgecolor('#1a1a1a')
    ax.zaxis.pane.set_edgecolor('#1a1a1a')
    ax.set_title(f'Action: {action_name}', color='white', fontsize=13, pad=10)

    # Compute axis limits from the entire sequence for stable view
    all_xyz = skeleton_seq.reshape(-1, 3)
    margin = 0.2
    x_lim = [all_xyz[:, 0].min() - margin, all_xyz[:, 0].max() + margin]
    y_lim = [all_xyz[:, 1].min() - margin, all_xyz[:, 1].max() + margin]
    z_lim = [all_xyz[:, 2].min() - margin, all_xyz[:, 2].max() + margin]

    def draw_frame(t):
        ax.cla()
        ax.set_facecolor('#0d0d0d')
        ax.set_title(f'Action: {action_name}  [frame {t+1}/{T}]', color='white', fontsize=11)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        # ax.view_init(elev=20, azim=90)
        ax.set_xlabel('X', color='#555555', fontsize=8)
        ax.set_ylabel('Z', color='#555555', fontsize=8)  # NTU: Y=up, Z=forward
        ax.set_zlabel('Y', color='#555555', fontsize=8)
        ax.tick_params(colors='#333333')

        joints = skeleton_seq[t]  # (25, 3)
        # Draw bones as lines
        for (i, j) in SKELETON_EDGES:
            xs = [joints[i, 0], joints[j, 0]]
            ys = [joints[i, 2], joints[j, 2]]  # Z as Y-axis (depth)
            zs = [joints[i, 1], joints[j, 1]]  # Y as Z-axis (height)
            ax.plot(xs, ys, zs, color='#00e5ff', linewidth=2.0, alpha=0.85)

        # Draw joints as scatter dots
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color='#ff6b6b', s=20, zorder=5, alpha=0.9)

        # Highlight head and hands
        for special_j, color in [(3, '#ffeb3b'), (11, '#69f0ae'), (7, '#69f0ae')]:
            ax.scatter(joints[special_j, 0], joints[special_j, 2], joints[special_j, 1], color=color, s=60, zorder=6)

    # only render every other frame to keep file size down
    step = 2
    frames = list(range(0, T, step))
    interval = int(1000 / fps * step)  # ms between frames
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=frames,
        interval=interval, blit=False
    )

    # Save to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    tmp.close()
    anim.save(tmp.name, writer='pillow', fps=fps // step)

    plt.close(fig)
    
    return tmp.name