from configs.parameters_der6 import *
import numba as nb


@nb.jit
def der_fn(x, t):
    pass
    return 1


def main(args, sampling_time=0.1):
    # time points
    start = datetime.now()
    x = []
    t_pre = np.linspace(0, sampling_time * 10, 11)
    t = np.linspace(0, sampling_time * 30, 31)
    if args.mode == 'Vnom':
        x_pre = odeint(der_fn, x0, t_pre, atol=1e-10, rtol=1e-11, mxstep=5000)
    else:
        x_pre = odeint(der_fn, x_critic, t_pre, atol=1e-10, rtol=1e-11, mxstep=5000)
    x1 = (x_pre[-1]).tolist()
    x.append(x_pre.tolist())
    x = x[0]
    for step in range(20):
        steps = 11 + step
        if args.mode == 'Vnom':
            x_af = odeint(der_fn, x1,
                          np.array([(steps - 1) * sampling_time, steps * sampling_time]), atol=1e-10, rtol=1e-11,
                          mxstep=5000)
            x = x + [x_af[-1].tolist()]
            x1 = (x_af[-1]).tolist()
        else:
            x_af = odeint(der_fn, x1,
                          np.array([(steps - 1) * sampling_time, steps * sampling_time]), atol=1e-10, rtol=1e-11,
                          mxstep=5000)
            x = x + [x_af[-1].tolist()]
            x1 = (x_af[-1]).tolist()
    print(datetime.now() - start)
    # control output
    vbus = []
    PDG = []
    QDG = []
    w = []

    # control input
    wn = []
    vn = []

    x = np.array(x)
    reward = 0

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

        # control input
        wn.append((x[:, DER_num * 13 + lines_num * 2 + loads_num * 2 + j + 1]).tolist())
        vn.append((x[:, DER_num * 14 + lines_num * 2 + loads_num * 2 + j + 1]).tolist())

        for q in range(len(vbus[j][11:])):
            vi = vbus[j][11:][q]
            if vi >= 0.95 and vi <= 1.05:
                reward += 0.05 - np.abs(1 - vi)
            elif vi <= 0.8 or vi >= 1.25:
                reward += -20
            else:
                reward += - np.abs(1 - vi)
    print(reward)

    # subplot: https://matplotlib.org/3.1.3/gallery/pyplots/pyplot_scales.html#sphx-glr-gallery-pyplots-pyplot-scales-py
    plt.figure()

    plt.subplot(221)
    for a in range(DER_num):
        plt.plot(t, w[a], label='DER_id %s' % (a + 1))
    # plt.xlim(0, 6)
    plt.xlabel("time")
    # plt.legend()
    plt.title("DER Frequency")

    plt.subplot(222)
    plt.xlabel("time")
    plt.ylabel("ratio")
    plt.title("Active power ratio")
    for b in range(DER_num):
        plt.plot(t, PDG[b], label='DER_id %s' % (b + 1))
    # plt.legend()
    # plt.xlim(0, 6)

    plt.subplot(223)
    plt.xlabel("time")
    plt.ylabel("voltage")
    plt.title("DER Voltage")
    for c in range(DER_num):
        plt.plot(t, vbus[c], label='DER_id %s' % (c + 1))
    # plt.legend()
    # plt.xlim(0, 6)

    plt.subplot(224)
    plt.xlabel("time")
    plt.ylabel("ratio")
    plt.title("Reactive Power Ratio")
    for d in range(DER_num):
        plt.plot(t, QDG[d], label='DER_id %s' % (d + 1))
    # plt.legend()
    # plt.xlim(0, 6)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    plt.show()
    # plt.savefig(args.plot_dir + 'DER_' + str(args.num_DER) + '.png')

    plt.xlabel("time")
    plt.ylabel("voltage")
    plt.title("DER Voltage")
    for c in range(len(vbus)):
        plt.plot(t, vbus[c], label='DER_id %s' % (c + 1))
    plt.legend()
    # plt.xlim(0, 6)
    plt.show()
    # plt.savefig(args.plot_dir + 'DER_Voltage' + '.png')

    plt.xlabel("time")
    plt.ylabel("Input wn")
    plt.title("Secondary frequency Control Input")
    for e in range(DER_num):
        plt.plot(t, wn[e], label='DER_id %s' % (e + 1))
    plt.legend()
    # plt.xlim(0, 6)
    plt.show()
    # plt.savefig(args.plot_dir + 'DER_freq_Input' + '.png')

    plt.xlabel("time")
    plt.ylabel("Input vn")
    plt.title("Secondary Voltage Control Input")
    for f in range(DER_num):
        plt.plot(t, vn[f], label='DER_id %s' % (f + 1))
    plt.legend()
    # plt.xlim(0, 6)
    plt.show()
    # plt.savefig(args.plot_dir + 'DER_vol_Input' + '.png')

    return reward


def parse_args():
    default_config_dir = 'configs'
    plot_dir = '../results/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, required=False,
                        default=default_config_dir, help="experiment config dir")
    parser.add_argument('--num_DER', type=int, required=False,
                        default=6, help="number of DERs")
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


if __name__ == '__main__':
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    import argparse
    import os
    from datetime import datetime

    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    num_test = 1
    random_seed = 0
    reward_list = []
    from configs.parameters_der6 import *

    DER_num = 6
    lines_num = 10  # no line 7, 9, 11
    loads_num = 4
    sampling_time = 0.05
    for i in range(num_test):
        np.random.seed(random_seed)
        random_seed += 1
        reward = main(args, sampling_time)  # False True
        reward_list.append(reward)
    print(np.mean(reward_list))
