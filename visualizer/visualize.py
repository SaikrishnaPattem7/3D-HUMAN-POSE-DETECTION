import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3D projection
from matplotlib.animation import FuncAnimation

# COCO 17-joint skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 6),                   # shoulders
    (5, 11), (6, 12),         # torso sides
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

LEFT_JOINTS  = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_JOINTS = {2, 4, 6, 8, 10, 12, 14, 16}

# Consistent view angle for all rendered frames
_ELEV, _AZIM = 15, -70


def _draw_pose(ax, keypoints_3d, dark=False):
    """Draw bones and joint markers onto a 3D axis."""
    ax.cla()

    if dark:
        ax.set_facecolor('#111111')
        bone_center = '#888888'
        lw = 3
        js = 60
    else:
        bone_center = 'dimgray'
        lw = 2
        js = 40

    x, y, z = keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2]

    for i, j in COCO_SKELETON:
        if i in LEFT_JOINTS or j in LEFT_JOINTS:
            color = '#4FC3F7'   # light blue
        elif i in RIGHT_JOINTS or j in RIGHT_JOINTS:
            color = '#FF7043'   # orange-red
        else:
            color = bone_center
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color=color, linewidth=lw)

    left_idx   = list(LEFT_JOINTS)
    right_idx  = list(RIGHT_JOINTS)
    center_idx = [i for i in range(17) if i not in LEFT_JOINTS and i not in RIGHT_JOINTS]

    ax.scatter(x[left_idx],   y[left_idx],   z[left_idx],   c='#4FC3F7', s=js, zorder=5, depthshade=False)
    ax.scatter(x[right_idx],  y[right_idx],  z[right_idx],  c='#FF7043', s=js, zorder=5, depthshade=False)
    ax.scatter(x[center_idx], y[center_idx], z[center_idx], c='#EEEEEE' if dark else bone_center, s=js, zorder=5, depthshade=False)

    ax.view_init(elev=_ELEV, azim=_AZIM)
    ax.set_xlabel('X', labelpad=2)
    ax.set_ylabel('Y', labelpad=2)
    ax.set_zlabel('Z', labelpad=2)


def _style_dark_ax(ax, fig):
    """Apply dark theme styling to a 3D axis."""
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#666666', labelsize=6)
    ax.xaxis.label.set_color('#666666')
    ax.yaxis.label.set_color('#666666')
    ax.zaxis.label.set_color('#666666')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333333')
    ax.yaxis.pane.set_edgecolor('#333333')
    ax.zaxis.pane.set_edgecolor('#333333')
    ax.grid(True, color='#333333', linewidth=0.5)


def render_pose_frame(keypoints_3d, size=480):
    """
    Render a single 3D pose to a dark-themed BGR numpy image (size x size).
    Used for embedding the 3D view into a video frame.
    """
    import cv2
    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    _draw_pose(ax, keypoints_3d, dark=True)
    _style_dark_ax(ax, fig)
    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size, size))
    return img


def plot_pose_3d(keypoints_3d, title='3D Pose', save_path=None):
    """
    Plot a single 3D pose as an interactive rotatable stick figure.

    Args:
        keypoints_3d: np.array (17, 3)
        title:        plot title
        save_path:    if provided, save the figure to this path instead of showing
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    _draw_pose(ax, keypoints_3d)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def animate_sequence(keypoints_sequence, interval=50, title='3D Pose', save_path=None):
    """
    Animate a sequence of 3D poses as a rotatable stick figure video.

    Args:
        keypoints_sequence: np.array (T, 17, 3)
        interval:           milliseconds between frames
        title:              base title (frame number appended)
        save_path:          if provided, save animation as .gif or .mp4
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        _draw_pose(ax, keypoints_sequence[frame_idx])
        ax.set_title(f'{title}  [frame {frame_idx + 1}/{len(keypoints_sequence)}]')

    anim = FuncAnimation(fig, update, frames=len(keypoints_sequence),
                         interval=interval, blit=False)

    if save_path:
        save_path = str(save_path)
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            n = len(keypoints_sequence)
            sample_indices = np.linspace(0, n - 1, min(6, n), dtype=int)
            fig2, axes = plt.subplots(2, 3, figsize=(18, 12),
                                      subplot_kw={'projection': '3d'})
            for ax2, idx in zip(axes.flat, sample_indices):
                _draw_pose(ax2, keypoints_sequence[idx])
                ax2.set_title(f'Frame {idx}')
            plt.suptitle(title)
            plt.tight_layout()
            fig2.savefig(save_path, dpi=150)
            plt.close('all')
            print(f"Saved sample frames to {save_path}")
        else:
            anim.save(save_path)
            plt.close()
            print(f"Saved animation to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    return anim
