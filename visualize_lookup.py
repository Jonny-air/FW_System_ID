import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import matplotlib.cm as cm

#Is called by the generate lookup script to visualize the lookup table
plt.rcParams.update({'font.size': 13})

def visualize(L1, model_name, show_titles =False):
    fig, axs = plt.subplots(2, figsize=(7, 7))
    ax = axs[0]
    ax2 = axs[1]
    alpha = 1
    _correct = False #hacky
    model_name = model_name.split('_')[0]
    title = f"{model_name} trim-look-up table"


    L = L1[1:, 1:6]
    # find points on the initial map
    # plot all the points in fpa, va map in the lookup table:
    airsps = L[0]
    fpas = L[1] * 180 / np.pi
    throts = L[2]
    pitchs = L[3] * 180 / np.pi
    z_s = np.zeros_like(fpas)

    num = airsps.size

    cmap = cm.get_cmap('gist_rainbow')

    colors = np.empty((13, 4))
    for i in range(13):
        colors[i] = cmap(1. * i / 13)  # color will now be an RGBA tuple
        if _correct:
            colors[i] = [0, 0, 0, 1]

    # np.transpose(np.append(np.append([r], [g], axis=0), [b], axis=0))

    def get_point(i):
        if not _correct:
            if i == 0:
                return "Min Sink +"
            elif i == 1:
                return "Max Climb"
            elif i == 2:
                return "Cruise"
            elif i == 3:
                return "Max Cruise"
            elif i == 4:
                return "Max Sink +"
            elif i == 5:
                return "Min Sink"
            elif i == 6:
                return "Max Sink"
        else:
            return None

    # 1-3
    i = 1
    x = np.linspace(airsps[i], airsps[i + 2], 10)
    x2 = np.linspace(throts[i], throts[i + 2], 10)
    y = np.linspace(fpas[i], fpas[i + 2], 10)
    y2 = np.linspace(pitchs[i], pitchs[i + 2], 10)
    ax.plot(x, y, color=cols.to_rgb(colors[-i]), alpha=alpha)
    ax2.plot(x2, y2, color=cols.to_rgb(colors[-i]), alpha=alpha)

    # 2-4
    i = 2
    x = np.linspace(airsps[i], airsps[i + 2], 10)
    x2 = np.linspace(throts[i], throts[i + 2], 10)
    y = np.linspace(fpas[i], fpas[i + 2], 10)
    y2 = np.linspace(pitchs[i], pitchs[i + 2], 10)
    ax.plot(x, y, color=cols.to_rgb(colors[-i]), alpha=alpha)
    ax2.plot(x2, y2, color=cols.to_rgb(colors[-i]), alpha=alpha)

    # 2-5
    i = 2
    x = np.linspace(airsps[i], airsps[0], 10)
    x2 = np.linspace(throts[i], throts[0], 10)
    y = np.linspace(fpas[i], fpas[0], 10)
    y2 = np.linspace(pitchs[i], pitchs[0], 10)
    ax.plot(x, y, color=cols.to_rgb(colors[-i - 1]), alpha=alpha)
    ax2.plot(x2, y2, color=cols.to_rgb(colors[-i - 1]), alpha=alpha)

    for i in range(0, airsps.size, 1):
        for n in [-1, 1]:
            x = np.linspace(airsps[i], airsps[(i + n) % 5], 10)
            x2 = np.linspace(throts[i], throts[(i + n) % 5], 10)
            y = np.linspace(fpas[i], fpas[(i + n) % 5], 10)
            y2 = np.linspace(pitchs[i], pitchs[(i + n) % 5], 10)
            ax.plot(x, y, color=cols.to_rgb(colors[(2 * i + n)]), alpha=alpha)
            ax2.plot(x2, y2, color=cols.to_rgb(colors[(2 * i + n)]), alpha=alpha)
        ax.scatter(x[0], y[0], label=get_point(i), color=cols.to_rgb(colors[2 * i]), zorder=10, alpha=alpha)
        ax2.scatter(x2[0], y2[0], label=get_point(i), color=cols.to_rgb(colors[2 * i]), zorder=10, alpha=alpha)

    # min sink
    throt = L1[3, 0]
    pitch = L1[4, 0] * 180 / np.pi
    i = 5
    if _correct:
        color = 'black'
    else:
        color = 'green'
    ax2.scatter(throt, pitch, label=get_point(i), color=color, zorder=10, alpha=alpha)
    # max climb
    throt = L1[3, 6]
    pitch = L1[4, 6] * 180 / np.pi
    i = 6
    if _correct:
        color = 'black'
    else:
        color = 'blue'
    ax2.scatter(throt, pitch, label=get_point(i), color=color, zorder=10, alpha=alpha)

    alpha = 1
    _correct = False

    ax.set_xlabel(r"Airspeed: $v_A$ $[\frac{m}{s}]$")
    ax.set_ylabel(r"Flight path angle: $\gamma$ [deg]")
    ax2.set_xlabel(r"Throttle: $\delta_t$ [%]")
    ax2.set_ylabel(r"Pitch: $\theta$ [deg]")

    ax.grid()
    ax2.grid()
    ax2.legend(loc='best', fontsize=10)
    if show_titles: ax.set_title(title)
    plt.tight_layout()
    plt.show()

def compare(Ls, model_name, title = '', show_titles=False):
    fig, axs = plt.subplots(2, figsize=(7, 7))
    ax = axs[0]
    ax2 = axs[1]
    alpha = 0.6
    _correct = True

    for L1 in Ls:
        L = L1[1:, 1:6]
        model_name = model_name.split('_')[0]
        #find points on the initial map
            #plot all the points in fpa, va map in the lookup table:
        airsps = L[0]
        fpas = L[1]*180/np.pi
        throts = L[2]
        pitchs = L[3]*180/np.pi
        z_s = np.zeros_like(fpas)

        num = airsps.size

        cmap = cm.get_cmap('gist_rainbow')

        colors = np.empty((13, 4))
        for i in range(13):
            colors[i] = cmap(1.*i/13)  # color will now be an RGBA tuple
            if _correct:
                colors[i] = [0,0,0,1]
        #np.transpose(np.append(np.append([r], [g], axis=0), [b], axis=0))


        def get_point(i):
            if not _correct:
                if i == 0:
                    return "Min Sink +"
                elif i == 1:
                    return "Max Climb"
                elif i == 2:
                    return "Cruise"
                elif i == 3:
                    return "Max Cruise"
                elif i == 4:
                    return "Max Sink +"
                elif i == 5:
                    return "Min Sink"
                elif i == 6:
                    return "Max Sink"
            else:
                return None


        #1-3
        i = 1
        x = np.linspace(airsps[i], airsps[i+2], 10)
        x2 = np.linspace(throts[i], throts[i+2], 10)
        y = np.linspace(fpas[i], fpas[i+2], 10)
        y2 = np.linspace(pitchs[i], pitchs[i + 2], 10)
        ax.plot(x, y, color=cols.to_rgb(colors[-i]), alpha = alpha)
        ax2.plot(x2, y2, color=cols.to_rgb(colors[-i]), alpha = alpha)

        #2-4
        i = 2
        x = np.linspace(airsps[i], airsps[i+2], 10)
        x2 = np.linspace(throts[i], throts[i+2], 10)
        y = np.linspace(fpas[i], fpas[i+2], 10)
        y2 = np.linspace(pitchs[i], pitchs[i + 2], 10)
        ax.plot(x, y, color=cols.to_rgb(colors[-i]), alpha = alpha)
        ax2.plot(x2, y2, color=cols.to_rgb(colors[-i]), alpha = alpha)

        #2-5
        i = 2
        x = np.linspace(airsps[i], airsps[0], 10)
        x2 = np.linspace(throts[i], throts[0], 10)
        y = np.linspace(fpas[i], fpas[0], 10)
        y2 = np.linspace(pitchs[i], pitchs[0], 10)
        ax.plot(x, y, color=cols.to_rgb(colors[-i-1]), alpha = alpha)
        ax2.plot(x2, y2, color=cols.to_rgb(colors[-i-1]), alpha = alpha)

        for i in range(0, airsps.size, 1):
            for n in [-1, 1]:
                x = np.linspace(airsps[i], airsps[(i+n)%5], 10)
                x2 = np.linspace(throts[i], throts[(i+n)%5], 10)
                y = np.linspace(fpas[i], fpas[(i+n)%5], 10)
                y2 = np.linspace(pitchs[i], pitchs[(i+n)%5], 10)
                ax.plot(x, y ,color=cols.to_rgb(colors[(2*i+n)]), alpha = alpha)
                ax2.plot(x2, y2, color=cols.to_rgb(colors[(2*i+n)]), alpha=alpha)
            ax.scatter(x[0], y[0], label=get_point(i), color=cols.to_rgb(colors[2*i]), zorder=10, alpha = alpha)
            ax2.scatter(x2[0], y2[0], label=get_point(i), color=cols.to_rgb(colors[2*i]), zorder=10, alpha = alpha)

        # min sink
        throt = L1[3, 0]
        pitch = L1[4, 0]*180/np.pi
        i = 5
        if _correct: color = 'black'
        else: color = 'green'
        ax2.scatter(throt, pitch, label=get_point(i), color=color,zorder=10, alpha = alpha)
        # max climb
        throt = L1[3, 6]
        pitch = L1[4, 6]*180/np.pi
        i = 6
        if _correct: color = 'black'
        else: color = 'blue'
        ax2.scatter(throt, pitch, label=get_point(i), color=color,zorder=10, alpha = alpha)

        alpha = 1
        _correct = False



    ax.set_xlabel(r"Airspeed: $v_A$ $[\frac{m}{s}]$")
    ax.set_ylabel(r"Flight path angle: $\gamma$ [deg]")
    ax2.set_xlabel(r"Throttle: $\delta_t$ [%]")
    ax2.set_ylabel(r"Pitch: $\theta$ [deg]")

    ax.grid()
    ax2.grid()
    ax2.legend(loc='best', fontsize=10)
    if show_titles: ax.set_title(title)
    plt.tight_layout()
    plt.show()
