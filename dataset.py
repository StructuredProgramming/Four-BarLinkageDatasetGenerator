#Created by the Department of Mechanical Engineering at Stony Brook University

import matplotlib.pyplot as plt
import scipy.spatial.distance as sciDist
import copy
import itertools
import numpy as np
import sys
import time
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def is_simple(tpMat):
    # simple: i.e., this can be solved with chain tree and intersection of circles/arc sects.
    # find links and set up joint table.
    # actuator will be noted with negative value.
    fixParam = [1, 3]
    jT = {}
    fixJ = []
    kkcJ = []
    chain = {}

    # step 1, initialize, set all joints and links to unknown (0 in jointTable) and jointLinkTable.
    for i in range(tpMat.shape[0]):
        jT[i] = 0
        chain[i] = {'from': None, 'next': []}

    # step 2, set all ground joints to known (1 to be known)
    for i in range(tpMat.shape[0]):
        if tpMat[i, i] in fixParam:
            jT[i] = 1
            fixJ.append(i)
            kkcJ.append((i, 'fixed', i))
            chain[i]['from'] = i

    # step 3, set joints in the kinematic chain to known
    pivotJ = fixJ
    while True:
        prevCtr = len(kkcJ)
        newJ = []
        for i in pivotJ:
            for j in range(tpMat.shape[1]):
                if tpMat[i, j] < 0 and jT[j] == 0:
                    jT[j] = 1
                    newJ.append(j)
                    kkcJ.append((j, 'chain', i))
                    chain[i]['next'].append(j)
                    chain[j] = {'from': i, 'next': []}

        if len(kkcJ) == prevCtr:
            break
        else:
            pivotJ = newJ  # This is based on the idea of tree node expansion

    if len(kkcJ) == tpMat.shape[0]:
        print(jT)
        return kkcJ, chain, True

    # step 4, set joints that can be solved through the intersection of circles to known
    while True:
        foundNew = False
        for k in jT:
            if jT[k] == 0:
                for i, _, _ in kkcJ:
                    for j, _, _ in kkcJ:
                        if i < j and tpMat[i, k] * tpMat[j, k] != 0 and not foundNew:
                            foundNew = True
                            jT[k] = 1
                            kkcJ.append((k, 'arcSect', (i, j)))
        if not foundNew:
            break

    # return chain and isSimple (meaning you can solve this with direct chain)
    return kkcJ, chain, len(kkcJ) == tpMat.shape[0]


# Direct kinematics:
def compute_chain_by_step(step, rMat, pos_init, unitConvert=np.pi / 180):
    pos_new = copy.copy(pos_init)
    dest, _, root = step
    pos_new[dest, 2] = rMat[root, dest] * unitConvert + pos_new[root, 2]
    c = np.cos(pos_new[dest, 2])
    s = np.sin(pos_new[dest, 2])
    posVect = pos_init[dest, 0:2] - pos_init[root, 0:2]
    pos_new[dest, 0] = posVect[0] * c - posVect[1] * s + pos_new[root, 0]
    pos_new[dest, 1] = posVect[0] * s + posVect[1] * c + pos_new[root, 1]
    return pos_new


# Inverse kinematics:
def compute_arc_sect_by_step(step, posOld, distMat, Ppp=None, threshold=0.1, timefactor=0.1):
    global is_impossible

    threshold = np.max(distMat) * threshold
    posNew = copy.copy(posOld)
    ptSect, _, centers = step
    cntr1, cntr2 = centers
    r1s = distMat[cntr1, ptSect]
    r2s = distMat[cntr2, ptSect]
    if r1s < 10e-12:
        posNew[ptSect, 0:2] = posOld[cntr1, 0:2]
    elif r2s < 10e-12:
        posNew[ptSect, 0:2] = posOld[cntr2, 0:2]
    else:
        ptOld = posOld[ptSect, 0:2]
        ptCen1 = posOld[cntr1, 0:2]
        ptCen2 = posOld[cntr2, 0:2]
        d12 = np.linalg.norm(ptCen1 - ptCen2)
        if d12 > r1s + r2s or d12 < np.absolute(r1s - r2s):
            # print('impossible \n')
            return posOld, False
        elif d12 < 10e-12:  # incidence joint
            # print('illegal \n')
            return posOld, False
        else:
            # print('legal')
            # a means the LENGTH from cntr1 to the mid point between two intersection points.
            # h means the LENGTH from the mid point to either of the two intersection points.
            # v means the Vector from cntr1 to the mid point between two intersection points.
            # vT 90 deg rotation of v
            a = (r1s ** 2 - r2s ** 2 + d12 ** 2) / (d12 * 2)
            h = np.sqrt(r1s ** 2 - a ** 2)
            v = ptCen2 - ptCen1
            vT = np.array([-v[1], v[0]])
            r1 = a / d12
            r2 = h / d12
            ptMid = ptCen1 + v * r1
            sol1 = ptMid + vT * r2
            sol2 = ptMid - vT * r2
            # print(ptOld, sol1, np.linalg.norm(sol1 - ptOld), sol2, np.linalg.norm(sol2 - ptOld))
            # compute ref point
            refPoint = ptOld
            if type(Ppp) != type(None):
                refPoint += (Ppp[ptSect, 0:2] - ptOld) * timefactor
            if np.linalg.norm(sol1 - refPoint) > np.linalg.norm(sol2 - refPoint):
                posNew[ptSect, 0:2] = sol2
                # print('sol2 selected \n')
            else:
                posNew[ptSect, 0:2] = sol1
            # detect if there's an abrupt change:
            if np.max(np.linalg.norm(posNew - posOld, axis=1)) > threshold:
                # print('thresholded', posNew, posOld)
                return posOld, False

        return posNew, True


# Basic data for computing a mechanism.
def compute_dist_mat(tpMat, pos):
    cdist = sciDist.cdist
    tpMat = copy.copy(np.absolute(tpMat))
    tpMat[list(range(0, tpMat.shape[0])), list(range(0, tpMat.shape[1]))] = 0
    return np.multiply(cdist(pos[:, 0:2], pos[:, 0:2]), tpMat)


def compute_curve_simple(tpMat, pos_init, rMat, distMat=None, maxTicks=360, baseSpeed=1):
    # preps
    kkcJ, chain, isReallySimple = is_simple(tpMat)
    if distMat is None:
        distMat = compute_dist_mat(tpMat, pos_init)
    poses1 = np.zeros((pos_init.shape[0], maxTicks, 3))
    poses2 = np.zeros((pos_init.shape[0], maxTicks, 3))
    # Set first tick
    poses1[:, 0, 0:pos_init.shape[1]] = pos_init
    poses2[:, 0, 0:pos_init.shape[1]] = pos_init
    # Compute others by step.
    meetAnEnd = False
    meetTwoEnds = False
    tick = 0
    offset = 0
    while not meetTwoEnds:
        # get tick
        tick += 1
        if tick + offset >= maxTicks:
            poses = poses1  # never flips
            break
        # decide which direction to compute.
        if not meetAnEnd:
            time = 1 * baseSpeed
            pos = poses1[:, tick - 1, :]
            posp = poses1[:, tick - 1, :]
            if tick - 2 < 0:
                Ppp = None
            else:
                Ppp = poses1[:, tick - 2, :]
        else:
            time = 1 * baseSpeed * (-1)
            pos = poses2[:, tick - 1, :]
            posp = poses2[:, tick - 1, :]
            if tick - 2 < 0:
                Ppp = None
            else:
                Ppp = poses2[:, tick - 2, :]
        # step-wise switch solution
        for step in kkcJ:
            if step[1] == 'fixed':
                pos[step[0], 0:2] = pos_init[step[0], :]
                notMeetEnd = True
            elif step[1] == 'chain':
                pos = compute_chain_by_step(step, rMat * time, posp[:, :])
                notMeetEnd = True
            elif step[1] == 'arcSect':
                if not meetAnEnd:
                    pos[step[0], :] = poses1[step[0], tick - 1, :]
                else:
                    pos[step[0], :] = poses2[step[0], tick - 1, :]
                pos, notMeetEnd = compute_arc_sect_by_step(step, pos, distMat, Ppp)
                if notMeetEnd and not meetAnEnd:  # never met an end -> to poses1
                    poses1[:, tick, :] = pos
                elif not notMeetEnd and not meetAnEnd:  # meet end like right now. This tick is not a solution.
                    poses1 = poses1[:, 0:tick, :]
                    offset = tick - 1  # the number of valid ticks
                    meetAnEnd = True
                    tick = 0  # reset tick to zero for time pos
                    break
                elif notMeetEnd and meetAnEnd:  # met an end. -> to poses2
                    poses2[:, tick, :] = pos
                else:  # not notMeetEnd and meetAnEnd. met both ends right now, this tick is not a solution.
                    poses2 = poses2[:, 1:tick, :]  # poses2 (<-) poses1(->). First pose of poses2 is pos_init.
                    poses2 = np.flip(poses2, axis=1)  # make poses2 (->)
                    poses = np.concatenate([poses2, poses1], axis=1)
                    meetTwoEnds = True
                    break
            else:
                print('Unexpected step:, ' + step[1])
                break
    return poses, meetAnEnd, isReallySimple


def get_pca_inclination(qx, qy):
    cx = np.mean(qx)
    cy = np.mean(qy)
    inf = False
    covar_xx = np.sum((qx - cx) * (qx - cx)) / len(qx)
    covar_xy = np.sum((qx - cx) * (qy - cy)) / len(qx)
    covar_yx = np.sum((qy - cy) * (qx - cx)) / len(qx)
    covar_yy = np.sum((qy - cy) * (qy - cy)) / len(qx)
    if np.isnan(covar_xx) or np.isnan(covar_yy) or np.isnan(covar_yx) or np.isnan(covar_xy)\
            or np.isinf(covar_xx) or np.isinf(covar_yy) or np.isinf(covar_yx) or np.isinf(covar_xy):
        inf = True
        phi = 0
    else:
        covar = np.array([[covar_xx, covar_xy], [covar_yx, covar_yy]])

        eig_val, eig_vec = np.linalg.eig(covar)

        # Inclination of major principal axis w.r.t. x axis
        if eig_val[0] > eig_val[1]:
            phi = np.arctan2(eig_vec[1, 0], eig_vec[0, 0])
        else:
            phi = np.arctan2(eig_vec[1, 1], eig_vec[0, 1])

    return phi, inf


def rotate_curve(x, y, theta):
    cpx = x * np.cos(theta) - y * np.sin(theta)
    cpy = x * np.sin(theta) + y * np.cos(theta)
    return cpx, cpy


start = time.time()
grid = np.arange(-1.0, 1.25, 0.25)

all_combinations = list(itertools.product(grid, repeat=2))  # doing all combinations of (-1, -1) ... (1, 1)

four_bars = list(itertools.product(all_combinations, repeat=2))  # we need only locations of moving pivots, so now 
# I do all possible combinations of [(-1, -1), (-1, -1), (-1, -1), ... (1, 1), (1, 1), (1, 1)]
print('Initial four-bar count - {}.'.format(len(four_bars)))
filtered_four_bars = []

# We do not want to have joints with the same location, for example two joints at (0, 0) cuz that will result in a non
# valid mechanism, so now I filter all of those combinations to remove such cases

for four_bar in four_bars:
    j_1, j_2 = four_bar
    j_0, j_3 = (0.0, 0.0), (1.0, 0.0)

    if j_0 == j_1 or j_0 == j_2 or j_0 == j_3 or j_1 == j_2 or j_1 == j_3 or j_2 == j_3:
        continue
    else:
        filtered_four_bars.append((j_0, j_1, j_2, j_3))

print('Four-bar count after removing same-joint ones - {}.'.format(len(filtered_four_bars)))

four_bars = dict()
# In the paper I shared with you, it says that if the ratio of moving links to its fixed link is of two different 
# mechanisms is the same, they will result in the same coupler curve, so we remove these mechanisms too, remember that 
# the length of our fixed link is equal to 1 so I do not even divide l_1, l_2, l_3 by l_4 (fixed link) since it is 
# equal to 1

for filtered_four_bar in filtered_four_bars:
    j_0, j_1, j_2, j_3 = filtered_four_bar

    l_1 = np.linalg.norm(np.array(j_1) - np.array(j_0))
    l_2 = np.linalg.norm(np.array(j_2) - np.array(j_1))
    l_3 = np.linalg.norm(np.array(j_3) - np.array(j_2))

    four_bars[(round(l_1, 6), round(l_2, 6), round(l_3, 6))] = filtered_four_bar

filtered_four_bars = list(four_bars.values())
print('Four-bar count after removing similar ones - {}.'.format(len(filtered_four_bars)))
four_bars = []

# Here I remove non-Grashof linkages 

for filtered_four_bar in filtered_four_bars:
    j_0, j_1, j_2, j_3 = filtered_four_bar

    l_1 = np.linalg.norm(np.array(j_1) - np.array(j_0))
    l_2 = np.linalg.norm(np.array(j_2) - np.array(j_1))
    l_3 = np.linalg.norm(np.array(j_3) - np.array(j_2))

    total = 1 + l_1 + l_2 + l_3

    shortest = min(1, l_1, l_2, l_3)
    longest = max(1, l_1, l_2, l_3)
    pq = total - shortest - longest

    if shortest + longest <= pq:
        four_bars.append(filtered_four_bar)

print('Four-bar count after removing non-Grashof ones - {}.'.format(len(four_bars)))
print('Four-bar count in total - {}.'.format(len(four_bars) * (len(all_combinations) - 5)))

# Here I finally get the coupler curves 
for four_bar in four_bars:
    j_0, j_1, j_2, j_3 = four_bar

    l_1 = np.linalg.norm(np.array(j_1) - np.array(j_0))
    l_2 = np.linalg.norm(np.array(j_2) - np.array(j_1))
    l_3 = np.linalg.norm(np.array(j_3) - np.array(j_2))
    
    if min(1, l_1, l_2, l_3) == 1 or min(1, l_1, l_2, l_3) == l_2:
        continue

    elif min(1, l_1, l_2, l_3) == l_1:
        tpTest = np.matrix([
            [1, -1, 1, 0, 0],
            [1, 2, 0, 1, 1],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 2, 1],
            [0, 1, 0, 1, 2]
        ])

        rMatTest = np.array([[0, 1.0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])
    else:
        tpTest = np.matrix([
            [1, 1, 1, 0, 0],
            [1, 2, 0, 1, 1],
            [1, 0, 1, -1, 0],
            [0, 1, 1, 2, 1],
            [0, 1, 0, 1, 2]
        ])

        rMatTest = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 1.0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])

    for j_4 in all_combinations:
        if j_4 in [j_0, j_1, j_2, j_3]:  # or j_4 == j_1 or j_4 == j_2 or j_4 == j_3:
            continue
        else:
            points_x, points_y = 0, 0

            posInit = np.array([j_0, j_1, j_3, j_2, j_4])
            jdxTest1, _, _ = compute_curve_simple(tpTest, posInit, rMatTest)

            if len(jdxTest1[4, :, 0]) < 360:
                continue

            points_x, points_y = np.asarray(jdxTest1[4, :, 0], dtype=np.float32), np.asarray(jdxTest1[4, :, 1])

            points_x, points_y = np.subtract(points_x, np.mean(points_x)), np.subtract(points_y, np.mean(points_y))

            points_x, points_y = np.divide(points_x, np.sqrt(np.var(points_x))), np.divide(points_y, np.sqrt(np.var(
                points_y)))

            # theta, is_inf = get_pca_inclination(np.asarray(points_x), np.asarray(points_y))
            #
            # if not is_inf:
            #     points_x, points_y = rotate_curve(points_x, points_y, -theta)
            #
            #     points_x, points_y = str(points_x).replace('\n', ''), str(points_y).replace('\n', '')
            #
            #     f.writelines('{} = {} = {} = {} = {} = {} = {}\n'.format(points_x, points_y, j_0, j_1, j_3, j_2,
            #                                                              j_4))
            plt.plot(points_x, points_y, color='black')
            plt.axis('equal')

            # moment = 10
            # plt.plot(jdxTest1[[0, 1], moment, 0], jdxTest1[[0, 1], moment, 1])
            # plt.plot(jdxTest1[[1, 4], moment, 0], jdxTest1[[1, 4], moment, 1])
            # plt.plot(jdxTest1[[2, 3], moment, 0], jdxTest1[[2, 3], moment, 1])
            # plt.plot(jdxTest1[[3, 4], moment, 0], jdxTest1[[3, 4], moment, 1])
            # plt.plot(jdxTest1[[3, 1], moment, 0], jdxTest1[[3, 1], moment, 1])
            # plt.show()

            plt.savefig('images\\{} {} {} {} {}.jpg'.format(j_0, j_1, j_3, j_2, j_4))
            plt.clf()

print(time.time()-start)
