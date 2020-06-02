"""""*******************************************************
 * Copyright (C) 2020 {Dong Chen} <{chendon9@msu.edu}>
 * DER graph generation function.
"""

import numpy as np
import math


class DER_Unit:
    def __init__(self, Rload, Lload, Kpv, Kiv, Kpc, Kic, mp, nq, rN, wc, F, wb, Lf, Cf, rLf, Lc, rLc):
        self.Rload = Rload
        self.Lload = Lload
        self.Kpv = Kpv
        self.Kiv = Kiv
        self.Kpc = Kpc
        self.Kic = Kic
        self.mp = mp
        self.nq = nq
        self.rN = rN
        self.wc = wc
        self.F = F
        self.wb = wb
        self.Lf = Lf
        self.Cf = Cf
        self.rLf = rLf
        self.Lc = Lc
        self.rLc = rLc

    def forward(self, x, wcom, wn, vn, I_ld, I_lq, disturbance_R, disturbance_L):
        Rload = self.Rload + disturbance_R * self.Rload
        Lload = self.Lload + disturbance_L * self.Lload
        # Rload = self.Rload
        # Lload = self.Lload
        # Transferring Inv. Output Currents to Global DQ
        ioD = math.cos(x[1]) * x[12] - math.sin(x[1]) * x[13]
        ioQ = math.sin(x[1]) * x[12] + math.cos(x[1]) * x[13]

        if Rload == 0:
            # Defining Bus Voltages
            vbD = self.rN * (ioD + sum(I_ld))
            vbQ = self.rN * (ioQ + sum(I_lq))
        else:
            vbD = self.rN * (ioD + sum(I_ld) - x[14])
            vbQ = self.rN * (ioQ + sum(I_lq) - x[15])

        # Transferring Bus Voltages to Inv. dq
        vbd = math.cos(x[1]) * vbD + math.sin(x[1]) * vbQ
        vbq = -math.sin(x[1]) * vbD + math.cos(x[1]) * vbQ

        # ------DG--------
        xdot1 = wn - self.mp * x[2] - wcom
        xdot2 = self.wc * (x[10] * x[12] + x[11] * x[13] - x[2])
        xdot3 = self.wc * (-x[10] * x[13] + x[11] * x[12] - x[3])
        vod_star = vn - self.nq * x[3]
        voq_star = 0
        xdot4 = vod_star - x[10]
        xdot5 = voq_star - x[11]
        ild_star = self.F * x[12] - self.wb * self.Cf * x[11] + self.Kpv * (vod_star - x[10]) + self.Kiv * x[4]
        ilq_star = self.F * x[13] + self.wb * self.Cf * x[10] + self.Kpv * (voq_star - x[11]) + self.Kiv * x[5]
        xdot6 = ild_star - x[8]
        xdot7 = ilq_star - x[9]
        vid_star = -self.wb * self.Lf * x[9] + self.Kpc * (ild_star - x[8]) + self.Kic * x[6]
        vid = vid_star
        viq_star = self.wb * self.Lf * x[8] + self.Kpc * (ilq_star - x[9]) + self.Kic * x[7]
        viq = viq_star
        xdot8 = (-self.rLf / self.Lf) * x[8] + wcom * x[9] + (1 / self.Lf) * (vid - x[10])
        xdot9 = (-self.rLf / self.Lf) * x[9] - wcom * x[8] + (1 / self.Lf) * (viq - x[11])
        xdot10 = wcom * x[11] + (1 / self.Cf) * (x[8] - x[12])
        xdot11 = -wcom * x[10] + (1 / self.Cf) * (x[9] - x[13])
        xdot12 = (-self.rLc / self.Lc) * x[12] + wcom * x[13] + (1 / self.Lc) * (x[10] - vbd)
        xdot13 = (-self.rLc / self.Lc) * x[13] - wcom * x[12] + (1 / self.Lc) * (x[11] - vbq)

        if Rload == 0:
            xdot14 = 0
            xdot15 = 0
        else:
            # ------Loads-------
            xdot14 = (-Rload / self.Lload) * x[14] + wcom * x[15] + (1 / Lload) * vbD
            xdot15 = (-Rload / self.Lload) * x[15] - wcom * x[14] + (1 / Lload) * vbQ

        return [xdot1, xdot2, xdot3, xdot4, xdot5, xdot6, xdot7, xdot8, xdot9, xdot10, xdot11, xdot12, xdot13, xdot14,
                xdot15], vbD, vbQ


class DER_controller:
    def __init__(self, mode, critic_bus_id, DER_num, lines_num, loads_num, DER_dic, BUSES, BUS_LOAD, rline, Lline,
                 a_ctrl, AP, G, Vnom,
                 wref, mp1, rN, wc, F, wb, Lf, Cf, rLf, Lc, rLc, kp, ki, random_init=True, sampling_time=0.1):
        self.DG = []
        self.wn_id_ls = []
        self.vn_id_ls = []
        self.vb_id_ls = []
        self.load_id_ls = []
        self.load_iq_ls = []
        self.load_count = 0
        self.DER_num = DER_num
        self.lines_num = lines_num
        self.loads_num = loads_num
        self.DER_dic = DER_dic
        self.BUSES = BUSES
        self.BUS_LOAD = BUS_LOAD
        self.rline = rline
        self.Lline = Lline
        self.Vnom = Vnom
        self.wref = wref
        self.a_ctrl = a_ctrl
        self.G = G
        self.AP = AP
        self.mp1 = mp1
        self.kp = kp
        self.ki = ki
        self.mode = mode
        self.critic_bus_id = critic_bus_id
        self.sampling_time = sampling_time
        self.disturbance_R = np.array([0] * self.DER_num)
        self.disturbance_L = np.array([0] * self.DER_num)

        for i in range(self.DER_num):
            if random_init:
                ratio_R = 0.4 * np.random.rand() + 0.8
                ratio_L = 0.4 * np.random.rand() + 0.8
            else:
                ratio_R = 1
                ratio_L = 1

            self.DG.append(DER_Unit(self.DER_dic[i][0]*ratio_R, self.DER_dic[i][1]*ratio_L, self.DER_dic[i][2], self.DER_dic[i][3],
                                    self.DER_dic[i][4],
                                    self.DER_dic[i][5], self.DER_dic[i][6], self.DER_dic[i][7], rN, wc, F, wb, Lf, Cf,
                                    rLf, Lc, rLc))
        if self.mode == 'Vnom':
            self.xdot = [0] + [0] * (self.DER_num * 15 + self.lines_num * 2 + self.loads_num * 2)
        else:
            self.xdot = [0] + [0] * (self.DER_num * 15 + self.lines_num * 2 + self.loads_num * 2 + 1)
        self.vbD = [0] * self.DER_num
        self.vbQ = [0] * self.DER_num

        # compute to determine the line id matrix
        self.line_id = [[0] * self.DER_num for i in range(self.DER_num)]
        self.line_count = 1  # start from 1
        for a in range(self.DER_num):
            index = [n for n, value in enumerate(self.BUSES[a]) if value == 1]
            index = [elem for elem in index if elem > a]  # only consider the upper triangular part of the matrix
            for j in range(len(index)):
                self.line_id[a][index[j]] = self.line_count
                self.line_count += 1

            # compute the indexes id for the secondary controller: wni and vni
            self.wn_id_ls.append(13 * self.DER_num + 2 * self.lines_num + 2 * self.loads_num + a + 1)
            self.vn_id_ls.append(14 * self.DER_num + 2 * self.lines_num + 2 * self.loads_num + a + 1)

            # compute load id list
            if self.BUS_LOAD[a] == 1:
                self.load_id_ls.append(13 * self.DER_num + 2 * self.lines_num + 2 * self.load_count + 1)
                self.load_iq_ls.append(13 * self.DER_num + 2 * self.lines_num + 2 * self.load_count + 2)
                self.load_count += 1
            else:
                self.load_id_ls.append(0)
                self.load_iq_ls.append(0)

        assert self.line_count == (self.lines_num + 1)
        assert self.load_count == self.loads_num

        # compute the line connection
        for line in range(self.lines_num):
            self.vb_id_ls.append(
                [[x, y] for x, li in enumerate(self.line_id) for y, val in enumerate(li) if val == line + 1])

    def VSI_VFctrl_func(self, x, t):
        wcom = x[self.wn_id_ls[0]] - self.mp1 * x[2]
        w_ls = []
        V_ls = []
        Pratio_ls = []
        Qratio_ls = []
        D_ls = [[0] * self.DER_num for i in range(self.DER_num)]

        for i in range(self.DER_num):
            line_ls_id = []
            line_ls_iq = []

            # compute the indexes for the line input
            index = [n for n, value in enumerate(self.BUSES[i]) if value == 1]
            for j in range(len(index)):
                if index[j] < i:
                    line_id = self.line_id[index[j]][i] - 1
                    line_ls_id.append(x[13 * self.DER_num + 1 + line_id * 2])
                    line_ls_iq.append(x[13 * self.DER_num + 2 + line_id * 2])
                else:
                    line_id = self.line_id[i][index[j]] - 1
                    line_ls_id.append(-x[13 * self.DER_num + 1 + line_id * 2])
                    line_ls_iq.append(-x[13 * self.DER_num + 2 + line_id * 2])

            # forward function to update the state function of each DER
            [self.xdot[1 + 13 * i], self.xdot[2 + 13 * i], self.xdot[3 + 13 * i], self.xdot[4 + 13 * i],
             self.xdot[5 + 13 * i], self.xdot[6 + 13 * i], self.xdot[7 + 13 * i], self.xdot[8 + 13 * i],
             self.xdot[9 + 13 * i], self.xdot[10 + 13 * i], self.xdot[11 + 13 * i], self.xdot[12 + 13 * i],
             self.xdot[13 + 13 * i], self.xdot[self.load_id_ls[i]], self.xdot[self.load_iq_ls[i]]], self.vbD[i], \
            self.vbQ[i] = \
                self.DG[i].forward(
                    [0, x[1 + 13 * i], x[2 + 13 * i], x[3 + 13 * i], x[4 + 13 * i], x[5 + 13 * i], x[6 + 13 * i],
                     x[7 + 13 * i], x[8 + 13 * i], x[9 + 13 * i], x[10 + 13 * i], x[11 + 13 * i],
                     x[12 + 13 * i], x[13 + 13 * i], x[self.load_id_ls[i]], x[self.load_iq_ls[i]]],
                    wcom, x[self.wn_id_ls[i]], x[self.vn_id_ls[i]], line_ls_id, line_ls_iq, self.disturbance_R[i], self.disturbance_L[i])

        # -------------------------lines------------------
        # ------line1--------
        for line in range(self.lines_num):
            self.xdot[13 * self.DER_num + line * 2 + 1] = (-self.rline[line] / self.Lline[line]) * x[
                13 * self.DER_num + line * 2 + 1] + wcom * x[13 * self.DER_num + line * 2 + 2] + (
                                                                  1 / self.Lline[line]) * (
                                                                  self.vbD[self.vb_id_ls[line][0][0]] - self.vbD[
                                                              self.vb_id_ls[line][0][1]])
            self.xdot[13 * self.DER_num + line * 2 + 2] = (-self.rline[line] / self.Lline[line]) * x[
                13 * self.DER_num + line * 2 + 2] - wcom * x[13 * self.DER_num + line * 2 + 1] + (
                                                                  1 / self.Lline[line]) * (
                                                                  self.vbQ[self.vb_id_ls[line][0][0]] - self.vbQ[
                                                              self.vb_id_ls[line][0][1]])

        # Controller Parameters
        for w in range(self.DER_num):
            w_ls.append([x[self.wn_id_ls[w]] - self.DER_dic[w][6] * x[
                2 + 13 * w]])
            V_ls.append([x[self.vn_id_ls[w]] -
                         self.DER_dic[w][7] * x[3 + 13 * w]])

            Pratio_ls.append([self.DER_dic[w][6] * x[2 + 13 * w]])
            Qratio_ls.append([self.DER_dic[w][7] * x[3 + 13 * w]])

        w_array = np.array(w_ls)
        V_array = np.array(V_ls)
        Pratio = np.array(Pratio_ls)
        Qratio = np.array(Qratio_ls)

        if self.mode == 'Vnom':
            Vref = self.Vnom
        else:
            # Critical Bus Voltage Control
            self.xdot[-1] = (self.Vnom - np.sqrt(
                (self.vbD[self.critic_bus_id - 1]) ** 2 + (self.vbQ[self.critic_bus_id - 1]) ** 2))
            Vref = self.kp * (self.Vnom - np.sqrt(
                (self.vbD[self.critic_bus_id - 1]) ** 2 + (self.vbQ[self.critic_bus_id - 1]) ** 2)) + self.ki * x[-1]

        if t < self.sampling_time * 10:
            self.xdot[(13 * self.DER_num + 2 * self.lines_num + 2 * self.loads_num + 1):] = [0] * (
                    len(self.xdot) - (13 * self.DER_num + 2 * self.lines_num + 2 * self.loads_num + 1))
        else:
            A = self.AP
            for d in range(self.DER_num):
                D_ls[d][d] = np.sum(A[d, :])

            D = np.array(D_ls)
            L = D - A

            Synch_Mat = -1 * self.a_ctrl * (
                    np.dot(L + self.G, (w_array - np.array([[self.wref]] * self.DER_num))) + np.dot(L,
                                                                                                    Pratio))
            Vol_Mat = -1 * self.a_ctrl * (
                    np.dot(L + self.G, (V_array - np.array([[Vref]] * self.DER_num))) + np.dot(L,
                                                                                               Qratio))
            for e in range(self.DER_num):
                self.xdot[self.wn_id_ls[e]] = Synch_Mat[e][0]
                self.xdot[self.vn_id_ls[e]] = Vol_Mat[e][0]
        # print(t)
        return self.xdot
