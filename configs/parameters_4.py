import numpy as np
import math

pi = math.pi

Cf = 50e-6
Lf = 1.35e-3
rLf = .1
Lc = .35e-3
rLc = .03
wc = 31.41

mp1 = 9.4 * 1e-5
nq1 = 1.3 * 1e-3
mp2 = mp1
nq2 = nq1
mp3 = 12.5 * 1e-5
nq3 = 1.5 * 1e-3
mp4 = mp3
nq4 = nq3

Kpv1 = 0.1
Kiv1 = 420
Kpv2 = Kpv1
Kiv2 = Kiv1
Kpc1 = 15
Kic1 = 20 * 1e3
Kpc2 = Kpc1
Kic2 = Kic1

Kpv3 = 0.05
Kiv3 = 390
Kpv4 = Kpv3
Kiv4 = Kiv3
Kpc3 = 10.5
Kic3 = 16 * 1e3
Kpc4 = Kpc3
Kic4 = Kic3

F = .75
rN = 1e4
wb = 2 * pi * 60
wref = 2 * pi * 60
Vnom = 380

# for critical bus control
kp = 4
ki = 40

# Network Data
rline1 = 0.23
Lline1 = 0.318 / (2 * pi * 60)
rline2 = 0.35
Lline2 = 1.847 / (2 * pi * 60)
rline3 = 0.23
Lline3 = 0.318 / (2 * pi * 60)

Rload1 = 12
Lload1 = 5 / (2 * pi * 60)
Rload2 = 15
Lload2 = 5 / (2 * pi * 60)

# Controller Parameters
a_ctrl = 40

AP = 4 * np.array([[0, 0, 0, 0],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   ])

# Pinning gain to the reference frequency
G = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# matrix for bus and load connection
BUS_LOAD = [1, 0, 1, 0]

# matrix for bus connection
BUSES = [[0, 1, 0, 0],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0]]

DER_dic = {
    0: [Rload1, Lload1, Kpv1, Kiv1, Kpc1, Kic1, mp1, nq1],
    1: [0, 0, Kpv2, Kiv2, Kpc2, Kic2, mp2, nq2],
    2: [Rload2, Lload2, Kpv3, Kiv3, Kpc3, Kic3, mp3, nq3],
    3: [0, 0, Kpv4, Kiv4, Kpc4, Kic4, mp4, nq4]
}

rline = [rline1, rline2, rline3]
Lline = [Lline1, Lline2, Lline3]

# add 0, to meet the difference between MATLAB and python, shape=(71,)
x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      2 * pi * 60, 2 * pi * 60, 2 * pi * 60, 2 * pi * 60,
      Vnom, Vnom, Vnom, Vnom]

x_critic = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, Vnom, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2 * pi * 60, 2 * pi * 60, 2 * pi * 60, 2 * pi * 60,
            Vnom, Vnom, Vnom, Vnom, 0]

mp = [mp1, mp2, mp3, mp4]
nq = [nq1, nq2, nq3, nq4]
