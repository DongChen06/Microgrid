"""""*******************************************************
 * Copyright (C) 2020 {Dong Chen} <{chendon9@msu.edu}>
 * main function.
"""

from scipy.integrate import odeint
import matplotlib.pyplot as plt
from DER_fn import DER_controller
import argparse
import os


def parse_args():
    default_config_dir = 'configs'
    plot_dir = 'results/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=False,
                        default=default_config_dir, help="experiment config dir")
    parser.add_argument('--num_DER', type=int, required=False,
                        default=4, help="number of DERs")
    parser.add_argument('--mode', type=str, required=False,
                        default='Vnom', help="voltage control mode", choices=['Vnom', 'Vcritc'])
    parser.add_argument('--critic_bus_id', type=int, required=False,
                        default=2, help="critical bus id")
    parser.add_argument('--plot_dir', type=str, required=False,
                        default=plot_dir, help="directory for storing results")
    parser.add_argument('--plot_unit_voltage', type=bool, required=False,
                        default=True, help="plot per unit voltage or not")

    args = parser.parse_args()
    return args


def main(args, DER_num, lines_num, loads_num, DER_controller):
    # time points
    t = np.linspace(0, 2)
    if args.mode == 'Vnom':
        x = odeint(DER_controller.VSI_VFctrl_func, x0, t, atol=1e-10, rtol=1e-11, mxstep=5000)
    else:
        x = odeint(DER_controller.VSI_VFctrl_func, x_critic, t, atol=1e-10, rtol=1e-11, mxstep=5000)
    vbus = []
    PDG = []
    QDG = []
    w = []
    x = np.array(x)
    # active/reactive power ratio
    for j in range(DER_num):
        PDG.append((mp[j] * (np.multiply(x[:, 13 * j + 10], x[:, 13 * j + 12]) + np.multiply(x[:, 13 * j + 11],
                                                                                             x[:,
                                                                                             13 * j + 13]))).tolist())
        QDG.append((nq[j] * (np.multiply(-x[:, 13 * j + 10], x[:, 13 * j + 13]) + np.multiply(x[:, 13 * j + 11],
                                                                                              x[:,
                                                                                              13 * j + 12]))).tolist())
        # frequency
        w.append(
            ((x[:, DER_num * 13 + lines_num * 2 + loads_num * 2 + j + 1] - mp[j] * x[:, 13 * j + 2]) / (
                    2 * pi)).tolist())
        # voltage of buses
        if args.plot_unit_voltage is True:
            vbus.append((np.sqrt(x[:, 13 * j + 10] ** 2 + x[:, 13 * j + 11] ** 2) / Vnom).tolist())
        else:
            vbus.append((np.sqrt(x[:, 13 * j + 10] ** 2 + x[:, 13 * j + 11] ** 2)).tolist())

    # subplot: https://matplotlib.org/3.1.3/gallery/pyplots/pyplot_scales.html#sphx-glr-gallery-pyplots-pyplot-scales-py
    plt.figure()

    plt.subplot(221)
    for a in range(DER_num):
        plt.plot(t, w[a], label='DER_id %s' % (a + 1))
    plt.xlim(0, 2)
    plt.xlabel("time")
    # plt.legend()
    plt.title("DER Frequency")

    plt.subplot(222)
    plt.xlabel("time")
    plt.ylabel("ratio")
    plt.title("Active power ratio")
    for b in range(DER_num):
        plt.plot(t, PDG[b], label='DER_id %s' % (b + 1))
    plt.legend()
    plt.xlim(0, 2)

    plt.subplot(223)
    plt.xlabel("time")
    plt.ylabel("voltage")
    plt.title("DER Voltage")
    for c in range(DER_num):
        plt.plot(t, vbus[c], label='DER_id %s' % (c + 1))
    # plt.legend()
    plt.xlim(0, 2)


    plt.subplot(224)
    plt.xlabel("time")
    plt.ylabel("ratio")
    plt.title("Reactive Power Ratio")
    for d in range(DER_num):
        plt.plot(t, QDG[d], label='DER_id %s' % (d + 1))
    # plt.legend()
    plt.xlim(0, 2)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    plt.show()
    # plt.savefig(args.plot_dir + 'DER_' + str(args.num_DER) + '.png')

    # plt.xlabel("time")
    # plt.ylabel("voltage")
    # plt.title("DER Voltage")
    # for c in range(len(vbus)):
    #     plt.plot(t, vbus[c], label='DER_id %s' % (c + 1))
    # plt.legend()
    # plt.xlim(0, 2)
    # plt.show()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    if args.num_DER == 4:
        from configs.parameters_4 import *
    else:
        from configs.parameters_20 import *

    DER_num = len(BUSES)
    lines_num = sum(sum(np.array(BUSES))) // 2
    loads_num = sum(BUS_LOAD)
    DER_controller = DER_controller(args.mode, args.critic_bus_id, DER_num, lines_num, loads_num, DER_dic, BUSES,
                                    BUS_LOAD, rline, Lline, a_ctrl, AP,
                                    G, Vnom, wref, mp1, rN, wc, F, wb, Lf, Cf, rLf, Lc, rLc, kp, ki)
    main(args, DER_num, lines_num, loads_num, DER_controller)
