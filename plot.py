import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

if __name__ == '__main__':
    xt = np.arange(0.1, 1.1, 0.1)
    freq = [1.00000000e+00, 1.00000000e+00, 2.00000000e+00, 3.00000000e+00,
            7.00000000e+00, 1.30000000e+01, 3.10000000e+01, 8.80000000e+01,
            3.32000000e+02, 2.71890000e+04]
    print(freq)

    fig = plt.figure(1, figsize=(40, 60))
    x_pos = np.arange(len(freq))
    plt.bar(x_pos, freq, align='center', alpha=0.5)
    plt.xticks(x_pos, xt)
    plt.ylabel('Occurrences')
    plt.title('Word Distribution')
    plt.show()
