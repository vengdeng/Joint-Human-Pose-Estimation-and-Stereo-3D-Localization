from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
COCO_PERSON_SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
        [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]]

def image_plot(pairs,img,name,depth_only):
    skeleton = []
    fig1 = plt.figure(figsize=(6, 8))
    #### image plot
    ################
    ax = fig1.add_subplot(2, 1, 1)
    iters2 = 0

    for t in pairs.keys():
        if pairs[t][-1] == -1:
            continue
        ### only perform for keypoints more than joints
        ## for very close part, the wrong association can have large disparity diffference
        depth = pairs[t][-1][1]
        sca_c = 'b'
        dist = pairs[t][-1][0]
        kp = pairs[t][0].reshape(1, -1)[0]
        sks = np.array(COCO_PERSON_SKELETON) - 1
        x = kp[0::3]
        y = kp[1::3]
        v = kp[2::3]
        for sk in sks:
            if np.all(v[sk] > 0):
                plt.plot(x[sk], y[sk], linewidth=0.3, color='r')
        plt.plot(x[v > 0], y[v > 0], 'o', markersize=2.5, markerfacecolor=sca_c, markeredgecolor='k', markeredgewidth=0.5)
        plt.plot(x[v > 1], y[v > 1], 'o', markersize=2.5, markerfacecolor=sca_c, markeredgecolor='k', markeredgewidth=0.5)
        if depth[depth > 0].var() > 200:
            continue
        if round(dist,1) < 12:
            plt.text(np.mean(x[x > 0]) - 10, y[y > 0].min() + 5 * (iters2 - 3), round(dist,1),
                     c='w', fontsize=10, weight="bold", bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        else:
            plt.text(np.mean(x[x > 0]) - 10, y[y > 0].min() + 5 * (iters2 - 3), round(dist,1),
                     c='w', fontsize=6, weight="bold", bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
        iters2 += 1
    ax.set_title('left stereo image')
    ax.imshow(img)
    #### skeleton plot
    ##################
    if not depth_only:
        ax = fig1.add_subplot(2, 1, 2, projection='3d')

        ax.set_zlabel('y')
        ax.set_ylabel('x')
        ax.set_xlabel('depth')
        ax.set_title('3D Pose')
        ax.set_ylim3d(0, img.shape[1])
        ax.set_zlim3d(0, img.shape[0])
        ax.set_xlim3d(0, 40)
        nums = []
        iters = 0
        for t in pairs.keys():
            if pairs[t][-1] == -1:
                continue
            value = pairs[t][0]
            value[:, 2] = pairs[t][-1][1]
            Z = np.copy(value)
            if len(Z[:, 2][Z[:, 2] > 0]) < 5:
                continue
            dist = pairs[t][-1][0]
            if dist[dist > 0].var() > 200:
                nums.append(iters)
                iters += 1
                continue
            Z[:, 2][abs(Z[:, 2] - np.median(Z[:, 2][Z[:, 2] > 0])) > 5] = 0
            Z1 = np.copy(Z[Z[:, 2] != 0])
            Z1[:, 2][Z1[:, 2] - np.median(Z1[:, 2][Z1[:, 2] > 0]) > 5] = 0
            if len(Z1) == 0:
                nums.append(iters)
            for i in COCO_PERSON_SKELETON:
                a, b = i
                if Z[a - 1, 2] == 0 or Z[b - 1, 2] == 0:
                    continue
                else:
                    skeleton.append([[Z[a - 1][2], Z[a - 1][0], Z[a - 1][1]], [Z[b - 1][2], Z[b - 1][0], Z[b - 1][1]]])

            iters += 1

            # plot sides
            ax.add_collection3d(Poly3DCollection(skeleton, edgecolors='r', alpha=.25))
            ax.scatter(Z1[:, 2], Z1[:, 0], Z1[:, 1], c='b')
        # rotate the axes and update
        for angle in range(0, 360):
            ax.view_init(210, angle)
    plt.savefig(name)
    fig1.canvas.draw()