import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams

from shapely.ops import unary_union

from matplotlib.path import Path

import matplotlib.patches as patches

import matplotlib.pyplot as pl
import pandas as pd
import numpy as np 
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import colorsys
import matplotlib.colors as mc

from matplotlib import rcParams
from scipy.spatial.distance import pdist

import powerlaw
from scipy.stats import pareto, norm, gaussian_kde, lognorm, kstest
from fincoretails import distributions, unipareto, lognormal, general_expareto,unipareto,general_powpareto,general_algpareto, loglikelihood_ratio, aic


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']
rcParams['font.size'] = 7

#colors = ["#1980E6", "#FF9123","#61C2FF" ]
colors = ["#D55E00", "#FF9123","#61C2FF",'#07AB92','#014C4C', '#FF7DBE']

cm = 1/2.54  # centimeters in inches


def load_data(name):
    df_mov = pd.read_csv(f'data/{name}_moving.csv')
    df_pdd = pd.read_csv(f'data/{name}_pdd.csv')
    df_ratio = pd.read_csv(f'data/{name}_ratio.csv')
    x_m = df_mov['x']
    y_m = df_mov['y']
    x_p = df_pdd['x']
    y_p = df_pdd['y']
    x_r = df_ratio['x']
    y_r = df_ratio['y']
    return x_m,y_m,x_p,y_p,x_r,y_r


def format_axes_log(ax):
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
    ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))    #ax.set_aspect('equal')

def compute_intercept(x_m,y_values,slope):
    intercept = np.log10(y_values) + slope*np.log10(x_m)    
    inter = np.mean(intercept)
    return inter


def darken_color(color, amount=0.4):
    """
    Darkens the given color by multiplying (1-luminosity) by the given amount.
    `color` can be matplotlib color string, hex color, or RGB tuple.

    Examples:
    >> darken_color('g', 0.3)
    >> darken_color('#F034A3', 0.6)
    >> darken_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

import matplotlib.colors as mcolors
def adjust_alpha(color, alpha=0.2):
    # Convert hex to RGB
    rgba = mcolors.to_rgba(color)
    rgb_hex = mcolors.to_hex(rgba[:3])
    alpha_hex = format(int(alpha * 255), '02x')
    return rgb_hex + alpha_hex