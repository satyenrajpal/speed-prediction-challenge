import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker


def moving_average(x, window):
  ret = np.zeros_like(x)

  for i in range(x.shape[0]):
    idx1 = max(0, i - (window - 1) // 2)
    idx2 = min(x.shape[0], i + (window - 1) // 2 + (2 - (window % 2)))

    ret[i] = np.mean(x[idx1:idx2])

  return ret

def plot(predict_values, gt):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(gt)), gt)
    ax.plot(np.arange(len(predict_values)), np.array(predict_values))
    start, end = ax.get_xlim()
    ax.yaxis.set_ticks(np.arange(0, max(gt) +10,  5.0))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    
    plt.xlabel('Frame num.')
    plt.ylabel('Speed [m/s]')
    # ax.figure.savefig('result.png', bbox_inches='tight')
    plt.show()