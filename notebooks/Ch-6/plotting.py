#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Various plotting methods for project.

Copyright (C) 2021 https://github.com/espenhgn

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""
import numpy as np
import scipy.signal as ss


# common plotting parameters
rcParams = {
    'axes.xmargin': 0.01,
    'axes.ymargin': 0.01,
    'font.size': 14,
    'legend.fontsize': 12,
    'axes.titlesize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 120.0,
    'axes.labelpad': 0,
    'legend.borderpad': 0.1,
    'legend.labelspacing': 0.1,
    'legend.framealpha': 1,
    # 'lines.linewidth': 2,
}

senkcolors = np.array([
    '#774411',   # L5E brown
    '#DDAA77',   # L5I
    '#771155',   # L6E pompadour
    '#CC99BB',   # L6I
    '#114477',   # L23E blue
    '#77AADD',   # L23I
    '#117744',   # L4E green
    '#88CCAA',   # L4I
    '#696969'])   # grayish


golden_ratio = (1 + np.sqrt(5)) / 2
figwidth = 14  # inches


def remove_axis_junk(ax, lines=['right', 'top']):
    """remove chosen lines from plotting axis"""
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        unit='mV',
        ylabels=True,
        color='r',
        ztransform=True,
        filter_data=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')):
    """helper function to draw line plots"""
    tvec = np.arange(data.shape[1]) * dt
    tinds = (tvec >= T[0]) & (tvec <= T[1])

    # apply temporal filter
    if filter_data:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    # subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch.%i' % (i + 1))
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[tinds][-1], tvec[tinds][-1]],
                [0.5, -0.5], lw=2, color='k', clip_on=False)
        # bbox = ax.get_window_extent().transformed(ax.get_figure().inverted())
        fig = ax.get_figure()
        figwidth = fig.figbbox.transformed(
            fig.dpi_scale_trans.inverted()).width
        axwidth = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()).width
        # bbox.width
        # ax.text(x[-1] + (x[-1] - x[0]) / width * 0.1, 0.5, 'test')
        ax.text(tvec[tinds][-1], 0,
        '\n\n$2^{' + '{}'.format(int(round(np.log2(vlimround)))
                                ) + '}$ ' + '{0}'.format(unit),
        color='k', rotation='vertical',
        va='center', ha='center')


    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r't (ms)', labelpad=0.1)

    return vlimround


def annotate_subplot(ax, ncols=1, nrows=1, letter='A',
                     linear_offset=0.1, fontsize=20,
                     fontweight='demibold'):
    '''add a subplot annotation'''
    ax.text(-ncols * linear_offset, 1 + nrows * linear_offset, letter,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize,
            fontweight=fontweight,
            transform=ax.transAxes)


def colorbar(fig, ax, im,
             width=0.01,
             height=1.0,
             hoffset=0.01,
             voffset=0.0,
             orientation='vertical'):
    '''
    draw matplotlib colorbar without resizing the axes object
    '''
    rect = np.array(ax.get_position().bounds)
    rect = np.array(ax.get_position().bounds)
    caxrect = [0] * 4
    caxrect[0] = rect[0] + rect[2] + hoffset * rect[2]
    caxrect[1] = rect[1] + voffset * rect[3]
    caxrect[2] = rect[2] * width
    caxrect[3] = rect[3] * height
    cax = fig.add_axes(caxrect)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    return cb
