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

from utils import *

#declare figure layout
fig = plt.figure(layout="constrained",figsize=(18.3*cm/2, 10*cm))

gs1 = GridSpec(2, 1, figure=fig)

ax1 = fig.add_subplot(gs1[0,0])
ax2 = fig.add_subplot(gs1[1, 0])

# gs4 = GridSpec(3, 1, figure=fig,left=0.42,right=0.6,wspace=0,hspace=0)

# ax9 = fig.add_subplot(gs4[0, 0])
# ax10 = fig.add_subplot(gs4[1, 0])
# ax11= fig.add_subplot(gs4[2, 0])


# gs2 = GridSpec(2, 2, figure=fig,left=0.62,hspace=0.2,bottom=0.6)

# ax3 = fig.add_subplot(gs2[0, 0])
# ax4 = fig.add_subplot(gs2[1, 0])
# ax5 = fig.add_subplot(gs2[0, 1])
# ax6 = fig.add_subplot(gs2[1, 1])

# gs3 = GridSpec(1, 2, figure=fig,left=0.62,top=0.45)
# ax7 = fig.add_subplot(gs3[0, 0])
# ax8 = fig.add_subplot(gs3[0, 1])

# gs5 = GridSpec(1, 1, figure=fig,left=0.62,bottom=0.1,hspace=0.2)
# ax_sim = fig.add_subplot(gs5[0, 0])



font_size = 11 

x_m,y_m,x_p,y_p,x_r,y_r = load_data('dk')

ax1.plot(x_m,y_m,'-',color=colors[1],label='observed movement prob.')
ax1.plot(x_p,y_p,'-',color=colors[2],label='pair distribution function')

ax1.set_ylim([0.8e-9,1.8e-2])
ax1.legend(loc='upper right',frameon=False)
format_axes_log(ax1)
ax1.set_xticks(np.logspace(1,6,6))
ax1.set_yticks(np.logspace(-8,-2,13))


ax2.plot(x_r,y_r,'-',color='k',label='intrinsic distance attractiveness')
slope= 0.98
intercept = y_r[0]/x_r[0]**(-slope)
ax2.plot(x_r,10**(slope*np.log10(x_r)+(intercept)),'--',color='k',lw=0.5,label='$y=x^{-s}$')
ax2.text(1e2,1e1,
        r' s$={:.4g}$'.format(slope),
        fontsize=6,ha='center')

format_axes_log(ax2)
ax2.set_xlabel('distance (m)')
ax2.set_ylabel('ratio',rotation=0, labelpad=0, va='top', ha='left')
ax2.yaxis.set_label_coords(0.03, 1)
ax2.legend(loc='upper right',frameon=False)
ax2.set_xticks(np.logspace(1,6,6))

ax1.set_ylabel('pdf',rotation=0, labelpad=0, va='top', ha='left')
ax1.yaxis.set_label_coords(0.03, 1)

ax1.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right')
ax2.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right')
ax1.xaxis.set_label_coords(1, -0.11)
ax2.xaxis.set_label_coords(1, -0.11)

ax1.tick_params(which='both', direction='in')
ax2.tick_params(which='both', direction='in')
ax1.set_xlim([10,1e6])
ax2.set_xlim([10,1e6])
ax1.set_ylim([1e-9,0.01])
ax2.set_ylim([1e-1,1e4])


# def city_plot(ax3,ax4,city_code= 84910847):
#     #load data 
#     df_piecewise = pd.read_csv(f'data/{city_code}_piecewise.csv')
#     df_piecewise_ratio = pd.read_csv(f'data/{city_code}_piecewise_ratio.csv')   
#     df_piecewise_fit = pd.read_csv(f'data/{city_code}_piecewise_fit.csv')
#     p = np.load(f'data/{city_code}_piecewise_params.npy')

#     xx = df_piecewise['x_m']
#     hist_mv = df_piecewise['y_m']
#     hist_pd = df_piecewise['y_p']
    
#     x_r = df_piecewise_ratio['x']
#     ratio = df_piecewise_ratio['y']
    
#     x_pw = df_piecewise_fit['x']
#     y_pw = df_piecewise_fit['y']
    
#     ms=1
    
#     ax3.plot(xx,hist_mv,'.',color=colors[1],label='movement distance',markersize=ms)
#     ax3.plot(xx,hist_pd,'.',color=colors[2],label='pair distribution function',markersize=ms)
    
#     ax4.axvline(x=10**p[0], color=colors[-1], linestyle=':',lw=2)
#     ax4.plot(x_pw, y_pw,'k--',alpha=0.5)
#     ax4.plot(x_r,ratio,'.',color='k',markersize=ms)

#     format_axes_log(ax3)
#     format_axes_log(ax4)
#     ax4.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right')
    
#     ax4.axvspan(0,10**p[0], color=colors[3],edgecolor=None, alpha=0.2,lw=0)
#     ax4.axvspan(10**p[0],1e6, color=colors[4],edgecolor=None, alpha=0.2,lw=0)

#     ax3.set_ylabel('pdf',rotation=0, labelpad=0, va='top', ha='right')
#     ax3.yaxis.set_label_coords(-0.05, 0.96)
#     ax4.set_ylabel('ratio',rotation=0, labelpad=0, va='top', ha='right')
#     ax4.yaxis.set_label_coords(-0.05, 0.96)
#     ax4.set_yticks([1e2,1e0])

#     ax4.xaxis.set_label_coords(1, -0.25)
#     ax4.set_xticks(np.logspace(1,6,6))
#     ax3.set_xticks(ax4.get_xticks())
#     ax3.set_xlim([10,1e6])
#     ax4.set_xlim([10,1e6])

#     ax3.tick_params(which='both', direction='in')  # 'both' applies the settings to both major and minor ticks
#     ax4.tick_params(which='both', direction='in')  # 'both' applies the settings to both major and minor ticks

    
# city_plot(ax3,ax4,city_code= 84910847)
# city_plot(ax5,ax6,city_code= 51018355)
# ax3.set_yticks([1e-5,1e-7])
# ax5.set_yticks([1e-6,1e-10])


# ### exponents distribution 

# data1 = np.load('data/pw-exponents_1.npy')
# data2 = np.load('data/pw-exponents-2.npy')
# print(f'n_city = {len(data1)}')

# kde1 = gaussian_kde(data1)
# kde2= gaussian_kde(data2)

# x1 = np.linspace(min(data1)-0.1, max(data1)+0.1, 100)
# x2 = np.linspace(min(data2), max(data2), 100)

# ax7.plot(-x1, kde1(x1), color=colors[3])
# ax7.fill_between(-x1, kde1(x1), color=colors[3], alpha=0.2)
# ax7.plot(-x2, kde2(x2), color=colors[4])
# ax7.fill_between(-x2, kde2(x2), color=colors[4], alpha=0.2)

# ax7.axvline(-np.mean(data1), color=colors[3], linestyle='--',lw=1)
# ax7.axvline(-np.mean(data2), color=colors[4], linestyle='--',lw=1)

# ax7.set_xticks([2,0.5])
# ax7.set_xticklabels(['2','0.5'])

# ax7.set_xlabel('exponent value',rotation=0, labelpad=0, va='top', ha='right')
# ax7.xaxis.set_label_coords(1, -0.2)
# ax7.spines['right'].set_visible(False)
# ax7.spines['top'].set_visible(False)
# ax7.set_ylim([0,2.2])
# ax7.set_xlim([-0.5,3])
# ax7.set_ylabel('pdf',rotation=0, labelpad=0, va='top', ha='right')
# ax7.yaxis.set_label_coords(-0.05, 0.98)
# ax7.set_yticklabels(['0','1',''])

# #ax8

# data = np.load('data/xmins.npy')
# print(f'n_city = {len(data)}')

# s = 0.954  # shape parameter

# bins_ln = np.logspace(2,5,20)
# shape, loc, scale = lognorm.fit(data, floc=0)
# hist_ln,bins_ln = np.histogram(data,bins_ln,density=True)
# xx = (bins_ln[1:]+bins_ln[:-1])/2
# pdf = lognorm.pdf(xx, shape, loc=loc, scale=scale)
# ax8.set_ylabel('pdf',rotation=0, labelpad=0, va='top', ha='right')
# ax8.yaxis.set_label_coords(-0.05, 0.98)

# ax8.set_xlabel('mobility city radius (m)',rotation=0, labelpad=0, va='top', ha='right')
# ax8.xaxis.set_label_coords(1,-0.2)
# ax8.tick_params(which='both', direction='in')

# fit = powerlaw.Fit(data)
# x = xx
# alpha = fit.power_law.alpha
# x_min = fit.power_law.xmin
# pdf_pl = x**(-alpha) / (alpha - 1) / x_min if alpha > 1 else np.zeros_like(x)


# ng = np.random.default_rng()

# # KS test vs power law
# ks, p = kstest(data, 'powerlaw', args=(alpha, x_min))
# print(f"KS test vs power law: {ks}, p-value: {p}")
# #test lognormal
# ks, p = kstest(data, 'lognorm', args=(shape, loc, scale))
# print(f"KS test vs lognormal: {ks}, p-value: {p}")

# mu,sigma = lognormal.fit_params(data)
# ax8.plot(xx,lognormal.pdf(xx, mu,sigma),'k--',lw=1,label='log-normal')
# alpha, xmin, beta = general_powpareto.fit_params(data)
# ax8.plot(xx,general_powpareto.pdf(xx, alpha, xmin, beta),'k:',lw=1,label='pow-Pareto')


# ax8.plot(xx,hist_ln,'.',color=colors[5],ms=5.5)

# ax8.legend(loc='lower left',frameon=False,fontsize=6)

# ax8.set_ylim([1e-7,1e-3])
# ax8.set_xlim([1e2,5e4])
# ax8.set_yticks([1e-6,1e-4])

# print(f'x_min {x_min}, alpha {alpha}, beta {beta}, mu {mu}, sigma {sigma}')
# format_axes_log(ax8)

# R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)


# print(f"\nComparison with Lognormal Distribution:")
# print(f"Likelihood ratio (R): {R}")
# print(f"p-value: {p}")

# #exp compare

# R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
# print(f"\nComparison with Exponential Distribution:")
# print(f"Likelihood ratio (R): {R}")
# print(f"p-value: {p}")

# #streched exp compare
# R, p = fit.distribution_compare('power_law', 'stretched_exponential', normalized_ratio=True)
# print(f"\nComparison with stretched Exponential Distribution:")
# print(f"Likelihood ratio (R): {R}")
# print(f"p-value: {p}")

# # Compare the fitted power-law distribution with a truncated power law (Pareto) distribution
# R, p = fit.distribution_compare('power_law', 'truncated_power_law')

# print(f"\nComparison with Truncated Power-law (Pareto) Distribution:")
# print(f"Likelihood ratio (R): {R}")
# print(f"p-value: {p}")

# # Compare the fitted power-law distribution with a truncated power law (Pareto) distribution
# R, p = fit.distribution_compare('lognormal', 'truncated_power_law')

# print(f"\nComparison with lognormal and Truncated Power-law (Pareto) Distribution:")
# print(f"Likelihood ratio (R): {R}")
# print(f"p-value: {p}")

# mu, sigma = lognormal.fit_params(data)

# Nsample = len(data)
# ll_gpp = general_powpareto.loglikelihoods(data, alpha, xmin, beta)
# ll_logn = lognormal.loglikelihoods(data, mu, sigma)
# logL_gpp = ll_gpp.sum()
# logL_logn = ll_logn.sum()

# R, p = loglikelihood_ratio(ll_gpp, ll_logn)
# AIC_gpp= aic(logL_gpp, number_of_free_parameters=3, nsamples=Nsample)
# AIC_logn = aic(logL_logn, number_of_free_parameters=2, nsamples=Nsample)



# print(f"gp pareto: {alpha=:4.2f}, {xmin=:4.2f}, {beta=:4.2f} ")
# print(f"lognormal: {mu=:4.2f}, {sigma=:4.2f}")
# print(f"logL gp pareto = {logL_gpp:4.2f}")
# print(f"logL lognormal = {logL_logn:4.2f}")
# print(f"log-likelihood ratio R={R:4.2f} with significance level p={p:4.2e}")
# print(f"AIC gp pareto = {AIC_gpp:4.2f}")
# print(f"AIC lognormal = {AIC_logn:4.2f}")
    



# # List of colors for elements and their fills

# darker_colors = [darken_color(color, 0.99) for color in colors]
# lighter_colors = [darken_color(color, 1.5) for color in colors]
# alpha_colors = [adjust_alpha(color, 1.5) for color in colors]

# # Parameters
# r1, r2, d = 1.2, 1.45, 6
# c1 = np.array([0, d/5])
# c2 = np.array([d, 0])

# def format_axes_geo(ax):
#     ax.axis('off')
#     ax.set_aspect('equal')
#     ax.set_xticks([])
#     ax.set_yticks([])

# def format_axes(ax):
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)

# # Ellipse parameters (a, b) computed from circle radii and centers distance
# a = 0.55 * d / 2 + max(r1, r2) + max(r1, r2)+0.3
# b = 0.6  * (np.sqrt(a**2 - (d/2)**2) + max(r1, r2)) +0.2

# # Center of the oval
# center = (c1 + c2) / 2

# # 4 control points for upper and lower half of the oval, respectively.
# upper_cp = np.array([(center[0] - a, center[1]), (center[0] - a, center[1] + b), 
#             (center[0] + a, center[1] + b), (center[0] + a, center[1])])
# lower_cp = np.array([(center[0] + a, center[1]), (center[0] + a, center[1] - b), 
#             (center[0] - a, center[1] - b), (center[0] - a, center[1])])

# # Angle of the line connecting the two cities
# angle = np.arctan2(c2[1] - c1[1], c2[0] - c1[0])

# # Rotation matrix
# rot = np.array([[np.cos(angle), -np.sin(angle)], 
#                 [np.sin(angle),  np.cos(angle)]])

# # Rotate control points
# upper_cp = np.dot(upper_cp - center, rot.T) + center
# lower_cp = np.dot(lower_cp - center, rot.T) + center

# # Introduce smooth noise
# np.random.seed(0)  
# for cp in [upper_cp, lower_cp]:
#     for i in range(len(cp)):
#         cp[i][0] += np.sin(cp[i][0]) * 0.25
#         cp[i][1] += np.sin(cp[i][1]) * 0.25

# codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
# upper_path = Path(upper_cp, codes)
# lower_path = Path(lower_cp, codes)

# format_axes(ax10)
# format_axes(ax11)
# # Add the paths
# patch_upper = patches.PathPatch(upper_path, facecolor= adjust_alpha(colors[4],0.2), lw=1.3, edgecolor=colors[4],zorder =1)
# patch_lower = patches.PathPatch(lower_path, facecolor=adjust_alpha(colors[4],0.2), lw=1.3, edgecolor=colors[4],zorder =1)

# ax9.add_patch(patch_upper)
# ax9.add_patch(patch_lower)
# # Cities as circles with borders
# city1 = plt.Circle(c1, r1, fill=True, facecolor='#FFFFFF', edgecolor=colors[3], lw=1.3,zorder =2)
# city2 = plt.Circle(c2, r2, fill=True, facecolor='#FFFFFF', edgecolor=colors[3], lw=1.3,zorder =2)
# ax9.add_patch(city1)
# ax9.add_patch(city2)
# city1 = plt.Circle(c1, r1, fill=True, facecolor=adjust_alpha(colors[3],0.2), edgecolor=None, lw=1.3,zorder =2)
# city2 = plt.Circle(c2, r2,  fill=True, facecolor=adjust_alpha(colors[3],0.2), edgecolor=None, lw=1.3,zorder =2)
# city1_e = plt.Circle(c1, r1, fill=False, edgecolor=colors[3], lw=1,zorder =4)

# city1_e = plt.Circle(c1, r1+r1/10,  fill=False,facecolor='none', edgecolor=colors[5],linestyle='--', lw=1.3,zorder =4)
# city2_e = plt.Circle(c2, r2+r2/10,  fill=False,facecolor='none', edgecolor=colors[5],linestyle='--', lw=1.3,zorder =4)

# # Add the cities
# ax9.add_patch(city1)
# ax9.add_patch(city2)
# ax9.add_patch(city1_e)
# ax9.add_patch(city2_e)

# # Add crosses at city centers
# ax9.plot(*c1, marker='+', markersize=1, color='k',zorder =5)
# ax9.plot(*c2, marker='+', markersize=1, color='k',zorder =5)

# angle = np.pi/6
# # Add city radius arrows
# offset = 0.15
# ax9.annotate("", xy=c1-offset, xytext=c1 + np.array([(r1+offset ) * np.cos(angle), (r1+offset) * np.sin(angle)]),
#             arrowprops=dict(arrowstyle="<-", color='k',lw=0.3),zorder =5)
# ax9.annotate("", xy=c2-offset, xytext=c2 + np.array([(r2+offset ) * np.cos(angle), (r2+offset ) * np.sin(angle)]),
#             arrowprops=dict(arrowstyle="<-", color='k',lw=0.3),zorder =5)

# # Add dashed lines from city centers to the edges of the 'd' line
# ax9.plot([c1[0], c1[0]], [c1[1], -2.8], color="black", linestyle="dashed",lw=0.1,zorder =5)
# ax9.plot([c2[0], c2[0]], [c2[1], -2.8], color="black", linestyle="dashed",lw=0.1,zorder =5)

# # Add random uniform dots inside city1
# N = 1200
# theta1 = np.random.uniform(0, 2 * np.pi, N)
# r1_uniform = np.sqrt(np.random.uniform(0, 1, N)) * r1
# x1 = c1[0] + r1_uniform * np.cos(theta1)
# y1 = c1[1] + r1_uniform * np.sin(theta1)

# # Generate random uniform dots inside city2
# theta2 = np.random.uniform(0, 2 * np.pi, N)
# r2_uniform = np.sqrt(np.random.uniform(0, 1, N)) * r2
# x2 = c2[0] + r2_uniform * np.cos(theta2)
# y2 = c2[1] + r2_uniform * np.sin(theta2)

# # Add distance arrow (lowered)
# ax9.annotate("", xy=(c1[0]-0.2, -2.8), xytext=(c2[0]+0.2, -2.8),
#             arrowprops=dict(arrowstyle="<->", color='k',lw=0.3),zorder =4)

# # Add text for city radius
# ax9.text(c1[0]+r1/3, c1[1] + r1/2, rf'$r_{1}$', va='center', ha='right',zorder =5)
# ax9.text(c2[0]+r2/3, c2[1] + r2/2, rf'$r_{2}$', va='center', ha='right',zorder =5)
# # Add text for distance
# ax9.text(center[0], c1[1] - r1 - 3.5, rf'$d$', va='center', ha='center',zorder =5)

# ax9.axis('equal') 
# ax9.set_xlim([-1.9,8])
# format_axes_geo(ax9)
# r1, r2, d = 1, 1.2, 30
# c1 = np.array([0, d/6])
# c2 = np.array([d, 0])

# # Add random uniform dots inside city1
# N1 = 1200
# theta1 = np.random.uniform(0, 2 * np.pi, N1)
# r1_uniform = np.sqrt(np.random.uniform(0, 1, N1)) * r1
# x1 = c1[0] + r1_uniform * np.cos(theta1)
# y1 = c1[1] + r1_uniform * np.sin(theta1)

# N2 = 2000
# # Generate random uniform dots inside city2
# theta2 = np.random.uniform(0, 2 * np.pi, N2)
# r2_uniform = np.sqrt(np.random.uniform(0, 1, N2)) * r2
# x2 = c2[0] + r2_uniform * np.cos(theta2)
# y2 = c2[1] + r2_uniform * np.sin(theta2)

# points1 = np.column_stack((x1, y1))
# points2 = np.column_stack((x2, y2))
# distances = pdist(np.concatenate((points1, points2),axis=0))

# max_x = 13*d/10
# bins = np.linspace(0,max_x,700)
# hist_h,bins = np.histogram(distances,bins)

# x = (bins[:-1]+bins[1:])/2
# ax10.plot(x,hist_h*1/x,color = colors[1],label='observed movement prob.')
# ax10.plot(x,hist_h,color = colors[2],label='pair distribution function')
# ax10.set_yscale('log')
# ax10.set_xscale('log')
# ax10.set_xlim([1,max_x])
# ax10.xaxis.set_minor_locator(plt.NullLocator())
# ax10.yaxis.set_minor_locator(plt.NullLocator())

# ax10.set_xticks([r1+0.2, r2+0.55, d])
# ax10.set_xticklabels([r'$r_1$', r'$r_2$', r'$d$'])
# N1N2 = 0.75e5
# n1n2d = N1N2/d
# ax10.set_yticks([N1N2,n1n2d])
# ax10.set_yticklabels([r'$N_1 N_2$', r'$\dfrac{N_1 N_2}{d}$'],fontsize=5)
# ax10.plot(x[x<d],[N1N2]*len(list(x[x<d])),'k--',lw=0.5)
# ax10.plot(x[x<d],[n1n2d]*len(list(x[x<d])),'k--',lw=0.5)
# ax10.legend(loc='upper right',frameon=False,fontsize=6)
# ax10.set_ylim([10,5e7])
# x = np.concatenate((np.array([0]),x),axis=0)
# hist_h = np.concatenate((np.array([0]),hist_h),axis=0)
# y = np.zeros(len(x))
# y[hist_h>0]= 1/x[hist_h>0]
# ax11.plot(x,y,zorder=2,color = 'k',label='intrinsic distance attractiveness')

# ax11.plot(x[x>0],1/x[x>0],'k--',lw=0.5,zorder=1,label='$y=x^{-1}$')
# ax11.set_yscale('log')
# ax11.set_xscale('log')
# ax11.set_xlim([1,max_x])
# ax11.set_ylim([0.001,10])
# # Remove other ticks and bars
# ax11.xaxis.set_minor_locator(plt.NullLocator())
# ax11.yaxis.set_minor_locator(plt.NullLocator())
# ax11.set_yticks([])
# ax11.set_yticklabels([])
# ax11.set_xticks([r1+0.2, r2+0.55, d])
# ax11.set_xticklabels([r'$r_1$', r'$r_2$', r'$d$'])
# ax11.legend(loc='upper right',frameon=False,fontsize=6)

# ax11.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right')
# ax11.xaxis.set_label_coords(1, -0.16)
# ax10.text(0.5, 1.05, r' $ P_{1 \to 2}=\frac{N_1 N_2}{d}$  ', transform=ax10.transAxes, ha='center', va='bottom',fontsize=8)

# ax7.tick_params(which='both', direction='in')  
# ax11.tick_params(which='both', direction='in')  
# ax10.tick_params(which='both', direction='in')  


# #simulation
# bins = np.logspace(1,5.7,100)
# x = (bins[:-1]+bins[1:])/2


# intrinsic_cost_sim = np.load('data/intrinsic_cost.npy')
# x_city = np.load('data/xmins.npy')


# ax_sim.axvspan(10,np.min(x_city), alpha=0.2, color=colors[3],lw=0) 
# ax_sim.axvspan(np.max(x_city),int(1e6), alpha=0.2, color=colors[4],lw=0)
# ax_sim.axvspan(np.min(x_city),np.max(x_city), alpha=0.1, color=colors[5],lw=0)
# ax_sim.vlines(np.min(x_city),0,200,color=colors[5],linestyle=':', lw = 2)
# ax_sim.vlines(np.max(x_city),0,200,color=colors[5],linestyle=':', lw = 2 )


# ax_sim.plot(x,intrinsic_cost_sim,'.',color='k',label='intrinsic moving distance',markersize=1)
# x_all = np.logspace(1,6.1,100)
# ax_sim.plot(x_all[:-1],x_all[:-1]**-1*intrinsic_cost_sim[0]*x_all[0]*2 ,'--',color='k',lw=0.5,label='$y=x^{-1}$')

# ax_sim.set_ylim([1e-4,2e2])
# ax_sim.set_xlim([10,1e6])
# format_axes_log(ax_sim)
# ax_sim.tick_params(which='both', direction='in')  
# ax_sim.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right',x=1,y=0.1)
# ax_sim.set_ylabel('ratio',rotation=0, labelpad=0, va='top', ha='left',x=0.1,y=1)
# ax_sim.set_yticks([1e-4,1e-2,1e0])

# ax8.set_yticks([1e-6,1e-5,1e-4])


# # figure layout 

gs1.tight_layout(fig)

# gs4.tight_layout(fig,pad=0.1)
# gs4.update(left=0.37,right=0.60,wspace=0.3, hspace=0.3)

# gs2.tight_layout(fig,pad=0.1)
# gs2.update(left=0.64,right=0.99,top=0.96,bottom=0.564,wspace=0.3, hspace=0.35)

# gs3.tight_layout(fig,pad=0.1)
# gs3.update(left=0.64,right=0.99,top=0.475,bottom=0.29,wspace=0.3, hspace=0.2)

# gs5.tight_layout(fig,pad=0.1)
# gs5.update(left=0.64,right=0.99,top=0.22,bottom=0.07,wspace=0.3, hspace=0.25)

# save
plt.tight_layout()
plt.savefig(f'figures/figure_3a.png',dpi=300)
plt.savefig(f'figures/figure_3a.pdf')

plt.close('all')