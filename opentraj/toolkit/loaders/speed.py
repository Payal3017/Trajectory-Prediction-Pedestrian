# Author: From TrajNet++
# Email: amiryan.j@gmail.com

from collections import namedtuple
import os
import numpy as np
import pandas as pd
import scipy.interpolate
from opentraj.toolkit.core.trajdataset import TrajDataset

# from opentraj.toolkit.core.trajdataset import TrajDataset

TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number', 'scene_id'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None, None)


# SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])

a = 39
wpp = [[] for _ in range(a)]
class CrowdLoader:
    def __init__(self, homog=[]):
        if len(homog):
            self.homog = homog
        else:
            self.homog = np.eye(3)

    def to_world_coord(self, loc):
        """Given H^-1 and world coordinates, returns (u, v) in image coordinates."""

        locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(self.homog, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2]

    def crowds_interpolate_person(self, ped_id, person_xyf):
        ## Earlier
        # xs = np.array([x for x, _, _ in person_xyf]) / 720 * 12 # 0.0167
        # ys = np.array([y for _, y, _ in person_xyf]) / 576 * 12 # 0.0208

        ## Pixel-to-meter scale conversion according to
        ## https://github.com/agrimgupta92/sgan/issues/5
        # xs_ = np.array([x for x, _, _ in person_xyf]) * 0.0210
        # ys_ = np.array([y for _, y, _ in person_xyf]) * 0.0239

        xys = self.to_world_coord(self.homog, np.array([[x, y] for x, y, _ in person_xyf]))
        xs, ys = xys[:, 0], xys[:, 1]

        fs = np.array([f for _, _, f in person_xyf])

        kind = 'linear'
        # if len(fs) > 5:
        #    kind = 'cubic'

        x_fn = scipy.interpolate.interp1d(fs, xs, kind=kind)
        y_fn = scipy.interpolate.interp1d(fs, ys, kind=kind)

        frames = np.arange(min(fs) // 10 * 10 + 10, max(fs), 10)

        return [TrackRow(int(f), ped_id, x, y)
                for x, y, f in np.stack([x_fn(frames), y_fn(frames), frames]).T]

    def load(self, filename):
        with open(filename) as annot_file:
            whole_file = annot_file.read()

            pedestrians = []
            current_pedestrian = []
            whole_person_path = {}



            ## Pixel-to-meter scale conversion according to
            ## https://github.com/agrimgupta92/sgan/issues/5
            # xs_ = np.array([x for x, _, _ in person_xyf]) * 0.0210
            # ys_ = np.array([y for _, y, _ in person_xyf]) * 0.0239
            # personKey=''
            for line in whole_file.split('\n'):
                if '- the number of splines' in line:
                    current_pedestrian = []
                    continue
                if '- Num of control points' in line:
                    if int(line[:2]):
                        personKey = int(line[:2])
                        if current_pedestrian:
                            pedestrians.append(current_pedestrian)
                        current_pedestrian = []
                        continue
                    if int(line[:1]):
                        personKey = int(line[:1])
                        if current_pedestrian:
                            pedestrians.append(current_pedestrian)
                        current_pedestrian = []
                        continue


                if ' - ' in line:
                    line = line[:line.find(' - ')]

                # tokenize
                entries = [e for e in line.split(' ') if e]
                if len(entries) != 4:
                    continue

                x, y, f, direction = entries
                # print('personKey ', personKey)
                wpp[personKey].append([float(x), float(y), int(f), float(direction)])

                # whole_person_path['personKey'].append(2)
                current_pedestrian.append([float(x), float(y), int(f)])
            # return wpp

        data = wpp
        length = data[4]
        start = 4
        end = 5
        pNumber = 4
        frame_number = 4553
        for i in range(start, end):
            if data[i]:
                for step in range(1, len(data[i])):
                    if data[pNumber][step][2] == frame_number:
                        a = np.array([[data[pNumber][step][0], data[pNumber][step][1]]])
                        print('a :', a)
                        b = np.array([[data[pNumber][step - 1][0], data[pNumber][step - 1][1]]])
                        print('a :', b)
                        travelled_FPS = data[pNumber][step][2] - data[pNumber][step - 1][2]
                        # pixwl to world coordinate conversion
                        a = self.to_world_coord(a)
                        b = self.to_world_coord(b)
                        dist = np.linalg.norm(a - b)
                        speed = abs(dist) / (travelled_FPS / 25)
                        print('travelled_FPS :', travelled_FPS)
                        print('speed --> 2D World Coordinate per second :', speed)
                sp = np.array([[data[pNumber][0][0], data[pNumber][0][1]]])
                print('sp :', sp)
                sp = self.to_world_coord(sp)

                dp = np.array([[data[pNumber][len(data[i]) - 1][0], data[pNumber][len(data[i]) - 1][1]]])
                dp = self.to_world_coord(dp)
                print('sp :', dp)
                distDes = np.linalg.norm(sp - dp)

                print('Distance from destination ((2D World coordinate)) :', distDes)

        start = 0
        end = 37

        p1 = 0
        p2 = 0
        NND = 0

        for i in range(start, end):
            if data[i]:
                current_dis = 0
                for j in range(2, end):
                    if data[j]:
                        for k in range(0, len(data[i]) - 1):
                            for l in range(0, len(data[j]) - 1):
                                # print(" len(data[i])", i)
                                # print(" len(data[j])", j)
                                if i == j:
                                    continue
                                if data[i][k][2] == data[j][l][2]:
                                    a = np.array([[data[i][k][0], data[i][k][1]]])
                                    b = np.array([[data[j][l][0], data[j][l][1]]])
                                    a = self.to_world_coord(a)
                                    b = self.to_world_coord(b)
                                    NND = np.linalg.norm(a - b)
                                    # print('Distance at time step :', data[i][k][2])

                                    # print()
                                    if current_dis == 0:
                                        current_dis = NND
                                    if current_dis < NND:
                                        current_dis = NND
                                        p1 = i
                                        p2 = j
                if p1 > 2 and p2 > 2:
                    print(' Nearest Neighbour of person =' + str(p1 + 1) + ' is person = ' + str(
                        p2 + 1) + 'at a distance (2D World coordinate)',
                          NND)


def load_crowds(path, **kwargs):
    """
        Note: pass the homography matrix as well
        :param path: string, path to folder
    """

    homog_file = kwargs.get("homog_file", "")
    Homog = (np.loadtxt(homog_file)) if os.path.exists(homog_file) else np.eye(3)
    raw_dataset = pd.DataFrame()

    data = CrowdLoader(Homog).load(path)




# test
if __name__ == "__main__":
    import os, sys
    import matplotlib.pyplot as plt

    # OPENTRAJ_ROOT = sys.argv[1]

    OPENTRAJ_ROOT = 'C:/Users/adila/PycharmProjects/OpenTraj/'
    # Zara data
    # =================================
    zara_01_vsp = os.path.join(OPENTRAJ_ROOT, 'datasets/UCY/zara01/annotation.vsp')
    zara_hmg_file = os.path.join(OPENTRAJ_ROOT, 'datasets/UCY/zara01/H.txt')
    zara_01_ds = load_crowds(zara_01_vsp, use_kalman=False, homog_file=zara_hmg_file)

    # trajs = zara_01_ds.get_trajectories()
    # trajs = [g for _, g in trajs]
    #
    # trajs = zara_01_ds.data.groupby(["scene_id", "agent_id"])
    # trajs = [(scene_id, agent_id, tr) for (scene_id, agent_id), tr in trajs]
    #
    # samples = zara_01_ds.get_entries()
    # plt.scatter(samples["pos_x"], samples["pos_y"])
    # print(zara_01_ds.data.pos_x)
    # print(zara_01_ds.data.pos_y)
    # plt.show()
