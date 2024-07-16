from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

NTU_PAIRS = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)

def visualize_sequence(seq, connections, name):
    def update(frame):
        ax.clear()

        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])

        ax.view_init(-45, 20, 90)
        ax.set_box_aspect(aspect_ratio)

        x = seq[frame, :, 0]
        y = seq[frame, :, 1]
        z = seq[frame, :, 2]

        for connection in connections:
            start = seq[frame, connection[0] - 1, :]
            end = seq[frame, connection[1] - 1, :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]

            ax.plot(xs, ys, zs)
        ax.scatter(x, y, z)

    
    print(f"Number of frames: {seq.shape[0]}")

    min_x, min_y, min_z = np.min(seq, axis=(0, 1))
    max_x, max_y, max_z = np.max(seq, axis=(0, 1))

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    aspect_ratio = [x_range, y_range, z_range]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # create the animation
    ani = FuncAnimation(fig, update, frames=seq.shape[0], interval=60)
    ani.save(f'{name}.gif', writer='pillow')
    
    plt.close(fig)



def visualize_sequence_pairs(seq1, seq2, name, connections=NTU_PAIRS):
    def update(frame):
        for ax, seq, title, min_x, min_y, min_z, max_x, max_y, max_z, aspect_ratio in zip([ax1, ax2],
                                                                                          [seq1, seq2],
                                                                                          ['Before', "After"],
                                                                                          [min_x1, min_x2],
                                                                                          [min_y1, min_y2],
                                                                                          [min_z1, min_z2],
                                                                                          [max_x1, max_x2],
                                                                                          [max_y1, max_y2],
                                                                                          [max_z1, max_z2],
                                                                                          [aspect_ratio1, aspect_ratio2]):
            ax.clear()
            ax.set_title(title, pad=40)
            ax.set_xlim3d([min_x, max_x])
            ax.set_ylim3d([min_y, max_y])
            ax.set_zlim3d([min_z, max_z])
            ax.view_init(-45, 20, 90)
            ax.set_box_aspect(aspect_ratio)
            x = seq[frame, :, 0]
            y = seq[frame, :, 1]
            z = seq[frame, :, 2]

            for connection in connections:
                start = seq[frame, connection[0] - 1, :]
                end = seq[frame, connection[1] - 1, :]
                xs = [start[0], end[0]]
                ys = [start[1], end[1]]
                zs = [start[2], end[2]]

                ax.plot(xs, ys, zs)
            ax.scatter(x, y, z)

    

    seq1 = seq1.permute(3, 1, 2, 0)  # (C, T, V, M) -> (M, T, V, C)
    seq1 = seq1[0]  # First person
    seq1 = seq1.cpu().detach().numpy()
    seq2 = seq2.permute(3, 1, 2, 0)  # (C, T, V, M) -> (M, T, V, C)
    seq2 = seq2[0]  # First person
    seq2 = seq2.cpu().detach().numpy()

    print(f"[INFO] Visualizing sequence pairs with number of frames: {seq1.shape[0]}")

    min_x1, min_y1, min_z1 = np.min(seq1, axis=(0, 1))
    max_x1, max_y1, max_z1 = np.max(seq1, axis=(0, 1))

    x_range = max_x1 - min_x1
    y_range = max_y1 - min_y1
    z_range = max_z1 - min_z1
    aspect_ratio1 = [x_range, y_range, z_range]

    min_x2, min_y2, min_z2 = np.min(seq2, axis=(0, 1))
    max_x2, max_y2, max_z2 = np.max(seq2, axis=(0, 1))

    x_range = max_x2 - min_x2
    y_range = max_y2 - min_y2
    z_range = max_z2 - min_z2
    aspect_ratio2 = [x_range, y_range, z_range]


    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Before")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("After")
    plt.tight_layout()

    # create the animation
    ani = FuncAnimation(fig, update, frames=seq1.shape[0], interval=60)
    ani.save(f'{name}.gif', writer='pillow')
    
    plt.close(fig)
    print("[INFO] visualiziation is done")