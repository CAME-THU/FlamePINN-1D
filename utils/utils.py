import numpy as np


def efmt(num):
    """e format"""
    s = "{:.1e}".format(num)
    part1 = s[0:5] if s[0] == "-" else s[0:4]
    return part1 + str(int(s[-3:]))


def cal_stat(file_dir, n_cal=3):
    metrics = []
    for i in range(n_cal):
        f = open(file_dir + f"{i + 1}/" + "metrics.txt", "r")
        texts = f.readlines()
        l2re_str = texts[1][9:-1]
        l2re_str = l2re_str.split("%, ")[:-1]
        metric = [float(l2re_str[j]) for j in range(len(l2re_str))]

        last_line = texts[-1]
        if last_line != "\n":
            last_line = last_line.split(" = ")
            nu_are = last_line[-1][:-2]
            nu_are = abs(float(nu_are))
            metric.append(nu_are)
        metrics.append(metric)
        f.close()

    metrics = np.array(metrics)
    stat_mean = np.mean(metrics, axis=0)
    stat_std = np.std(metrics, axis=0)

    np.savetxt(file_dir + f"metrics_statistic_1-{n_cal}.txt",
               np.vstack([stat_mean, stat_std]),
               fmt="%.4f")


