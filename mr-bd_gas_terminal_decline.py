import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as t
import theano as th
import math
import time

file_path = r''
data = pd.read_csv(file_path, delimiter=',', index_col=0, header=0)

num_wells = len(data.index.unique())
mcmc_trials = 30000
burn = 10000
forecast_period = np.arange(0, 600, 1)
returned_months = np.arange(0, 600, 1)
forecasts = np.empty((num_wells, 5))
dmin = 0.06
min_prod = 300
all_qi_traces = np.empty((mcmc_trials-burn)*num_wells)
all_b_traces = np.empty((mcmc_trials-burn)*num_wells)
all_di_traces = np.empty((mcmc_trials-burn)*num_wells)
all_shift_month_traces = np.empty((mcmc_trials-burn)*num_wells)
all_shift_dt_traces = np.empty((mcmc_trials-burn)*num_wells)
all_q_traces = np.empty((mcmc_trials-burn)*num_wells)
start_shift = pd.Series(index=data.index.unique())
sampling_trials = 1000
well_aggregation = 8

def q_fcst_t(qi, b, di, forecast_period, returned_months, dmin, shift_month, shift_dt, min_prod):
    q = qi / np.power(1+b*di*forecast_period, 1/b)
    d = t.concatenate([t.arange(1, 2), t.extra_ops.diff(q)*-1 / q[:-1]])
    terminal_mask = d < dmin/12
    terminal_start = t.argmax(terminal_mask)
    terminal_period = t.arange(terminal_start, len(forecast_period), 1) - (terminal_start - 1)
    qi_exp = q[terminal_start-1]
    q_exp = qi_exp*np.exp(-dmin/12*terminal_period)
    q_arps = q[:terminal_start]
    forecast = t.concatenate([q_arps, q_exp])
    if t.gt(shift_month, 0):
        pre_shift = forecast[:shift_month]
        post_shift_period = t.arange(shift_month, len(forecast_period), 1) - (shift_month - 1)
        post_shift_qi = forecast[shift_month]
        post_shift = post_shift_qi*np.exp(-shift_dt/12*post_shift_period)
        forecast = t.concatenate([pre_shift, post_shift])
    min_prod_mask = forecast < min_prod
    min_prod_start = t.argmax(min_prod_mask)
    post_min_period = t.arange(min_prod_start, len(forecast_period), 1) - (min_prod_start - 1)
    pre_min = forecast[:min_prod_start]
    post_min = 0*post_min_period
    forecast = t.concatenate([pre_min, post_min])
    return forecast[returned_months]

def q_fcst(qi, b, di, forecast_period, returned_months, dmin, shift_month, shift_dt, min_prod):
    shift_month = int(shift_month)
    q = qi / np.power(1+b*di*forecast_period, 1/b)
    d = np.concatenate((np.arange(1, 2), np.divide(np.diff(q)*-1, q[:-1])))
    terminal_mask = d < dmin/12
    terminal_start = terminal_mask.argmax()
    terminal_period = np.arange(terminal_start, len(forecast_period), 1) - (terminal_start - 1)
    qi_exp = q[terminal_start-1]
    q_exp = qi_exp*np.exp(-dmin/12*terminal_period)
    q_arps = q[:terminal_start]
    forecast = np.concatenate([q_arps, q_exp])
    if shift_month > 0:
        pre_shift = forecast[:shift_month]
        post_shift_period = np.arange(shift_month, len(forecast_period), 1) - (shift_month - 1)
        post_shift_qi = forecast[shift_month]
        post_shift = post_shift_qi*np.exp(-shift_dt/12*post_shift_period)
        forecast = np.concatenate([pre_shift, post_shift])
    min_prod_mask = forecast < min_prod
    min_prod_start = min_prod_mask.argmax()
    post_min_period = len(forecast_period) - min_prod_start
    pre_min = forecast[:min_prod_start]
    post_min = np.zeros(post_min_period)
    forecast = np.concatenate([pre_min, post_min])
    return forecast[returned_months]

def loss_func(well_prod, qi, b, di, forecast_period, dmin, start_shift):
    q_est = q_fcst(qi, b, di, forecast_period, well_prod['MONTH'] >= start_shift, dmin)
    err1 = (well_prod['GAS'] - q_est).sum()
    err2 = abs(well_prod['GAS'] - q_est).sum()
    if err1 + err2 > 0:
        return (err1+err2)**2
    else:
        return err1+err2

def aggregation(sampling_trials=sampling_trials, well_aggregation=well_aggregation):
    qi_samples = np.empty(sampling_trials)
    b_samples = np.empty(sampling_trials)
    di_samples = np.empty(sampling_trials)
    q_samples = np.empty(sampling_trials)
    loss_samples = np.empty(sampling_trials)
    for trial in range(sampling_trials):
        qi_temp = np.empty(well_aggregation)
        b_temp = np.empty(well_aggregation)
        di_temp = np.empty(well_aggregation)
        q_temp = np.empty(well_aggregation)
        loss_temp = np.empty(well_aggregation)
        for well in range(well_aggregation):
            rand_idx = np.random.randint(0, high=len(all_qi_traces))
            propnum_idx = math.floor(rand_idx/(mcmc_trials-burn))
            propnum = start_shift.index[propnum_idx]
            start_shift_val = start_shift.iloc[propnum_idx]
            well_prod = data.loc[propnum]
            qi_temp[well] = all_qi_traces[rand_idx]
            b_temp[well] = all_b_traces[rand_idx]
            di_temp[well] = all_di_traces[rand_idx]
            q = q_fcst(all_qi_traces[rand_idx], all_b_traces[rand_idx], all_di_traces[rand_idx], forecast_period, returned_months, dmin)
            loss = loss_func(well_prod, qi_temp[well], b_temp[well], di_temp[well], forecast_period, dmin, start_shift_val)
            q_temp[well] = q.sum()
            loss_temp[well] = loss
        qi_samples[trial] = qi_temp.mean()
        b_samples[trial] = b_temp.mean()
        di_samples[trial] = di_temp.mean()
        q_samples[trial] = q_temp.mean()
        loss_samples[trial] = loss_temp.mean()
    f, axarr = plt.subplots(2, 2)
    sns.distplot(qi_samples, norm_hist=True, ax=axarr[0, 0], axlabel='qi')
    sns.distplot(b_samples, norm_hist=True, ax=axarr[0, 1], axlabel='b')
    sns.distplot(di_samples, norm_hist=True, ax=axarr[1, 0], axlabel='di')
    sns.distplot(q_samples, norm_hist=True, ax=axarr[1, 1], axlabel='Q')
    plt.show()

for i, propnum in enumerate(data.index.unique()):
    start_time = time.time()
    prod_raw = data[data.index == propnum]
    prod_max = prod_raw['GAS'].max()
    prod_max_month = prod_raw[prod_raw['GAS'] == prod_max]['MONTH'][0]
    ##rolling_median = prod_raw['GAS'].rolling(window=5, center=True).median()
    ##diff = abs(prod_raw['GAS'] - rolling_median)
    ##outliers = (diff / rolling_median > 50) & (prod_raw['MONTH'] > 10)
    ##prod_filtered = prod_raw[~outliers]
    ##prod_outliers = prod_raw[outliers]
    prod_filtered = prod_raw

    if prod_max_month > 6:
        prod_max_month = 1

    if prod_max_month > 0:
        prod_filtered = prod_filtered[prod_filtered['MONTH'] >= prod_max_month]
        prod_filtered.loc[:, ['MONTH']] = prod_filtered.loc[:, ['MONTH']] - prod_max_month

    prod_filtered = prod_filtered[prod_filtered.loc[:,'GAS'] !=0]

    print(i, 'of', len(data.index.unique()), propnum)
    ##print('max diff:', (diff / diff.std()).max())
    ##print('num outliers:', outliers.sum())
    print('max prod month:', prod_max_month)
    print('forecast months:', prod_filtered.shape[0])
    
    with pm.Model() as Arps:

        if prod_filtered.shape[0] == 1:
            qi = pm.Uniform('qi', 0, 450000)
            b = pm.Uniform('b', lower=0.8, upper=1.5)
            di = pm.Uniform('di', lower=0.01, upper=0.6)
        else:
            qi = pm.Uniform('qi', 0, 450000)
            b = pm.Uniform('b', lower=0.01, upper=2.0)
            di = pm.Uniform('di', lower=0.01, upper=1)
            shift_month = pm.DiscreteUniform('sm', lower=0, upper=prod_filtered.shape[0]-1)
            shift_dt = pm.Uniform('sdt', lower=0, upper=0.99)
        q = q_fcst_t(qi, b, di, forecast_period, prod_filtered['MONTH'].values, dmin, shift_month, shift_dt, min_prod)
        sd = pm.HalfNormal('sd', sd=0.1)

        obs = pm.StudentT('obs', nu=1, mu=q, lam=sd, observed=prod_filtered['GAS'].values)

        trace = pm.sample(mcmc_trials, step=pm.Metropolis(), progressbar=False)

        pm.traceplot(trace)
        save_path = propnum + ' Trace Plots'
        plt.savefig(save_path)
        plt.close()

        pm.autocorrplot(trace)
        save_path = propnum + ' Autocorrelation Plot'
        plt.savefig(save_path)
        plt.close()

        pm.plot_posterior(trace)
        save_path = propnum + ' Posterior Plot'
        plt.savefig(save_path)
        plt.close()

        forecast_period_max = forecast_period + prod_max_month
        forecast_period_trunc = forecast_period[:(len(forecast_period)-prod_max_month)]
        returned_months_trunc = returned_months[:(len(returned_months)-prod_max_month)]
        
        qi_trace = trace.qi[burn:][:]
        b_trace = trace.b[burn:][:]
        di_trace = trace.di[burn:][:]
        shift_month_trace = trace.sm[burn:][:]
        shift_dt_trace = trace.sdt[burn:][:]
        q_trace = np.empty(len(qi_trace))
        
        for j in range(len(qi_trace)):
            all_qi_traces[(i+1)*j] = qi_trace[j]
            all_b_traces[(i+1)*j] = b_trace[j]
            all_di_traces[(i+1)*j] = di_trace[j]
            all_shift_month_traces[(i+1)*j] = shift_month_trace[j]
            all_shift_dt_traces[(i+1)*j] = shift_dt_trace[j]
            q_trace[j] = q_fcst(qi_trace[j], b_trace[j], di_trace[j], forecast_period, returned_months_trunc, dmin, shift_month_trace[j], shift_dt_trace[j], min_prod).sum()
            all_q_traces[(i+1)*j] = q_trace[j]

        p10 = np.percentile(q_trace, [9, 11])
        p10_range = np.where((q_trace >= p10[0]) & (q_trace <= p10[1]))
        qi_p10 = qi_trace[p10_range].mean()
        b_p10 = b_trace[p10_range].mean()
        di_p10 = di_trace[p10_range].mean()
        shift_month_p10 = shift_month_trace[p10_range].mean()
        shift_dt_p10 = shift_dt_trace[p10_range].mean()
        q_p10 = q_fcst(qi_p10, b_p10, di_p10, forecast_period, returned_months_trunc, dmin, shift_month_p10, shift_dt_p10, min_prod)
        
        p50 = np.percentile(q_trace, [49, 51])
        p50_range = np.where((q_trace >= p50[0]) & (q_trace <= p50[1]))
        qi_p50 = qi_trace[p50_range].mean()
        b_p50 = b_trace[p50_range].mean()
        di_p50 = di_trace[p50_range].mean()
        shift_month_p50 = shift_month_trace[p50_range].mean()
        shift_dt_p50 = shift_dt_trace[p50_range].mean()
        q_p50 = q_fcst(qi_p50, b_p50, di_p50, forecast_period, returned_months_trunc, dmin, shift_month_p50, shift_dt_p50, min_prod)

        p90 = np.percentile(q_trace, [89, 91])
        p90_range = np.where((q_trace >= p90[0]) & (q_trace <= p90[1]))
        qi_p90 = qi_trace[p90_range].mean()
        b_p90 = b_trace[p90_range].mean()
        di_p90 = di_trace[p90_range].mean()
        shift_month_p90 = shift_month_trace[p90_range].mean()
        shift_dt_p90 = shift_dt_trace[p90_range].mean()
        q_p90 = q_fcst(qi_p90, b_p90, di_p90, forecast_period, returned_months_trunc, dmin, shift_month_p90, shift_dt_p90, min_prod)
            
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
        plt.plot(forecast_period[prod_max_month:], q_p50, sns.xkcd_rgb['denim blue'], alpha=0.75, lw=1, zorder=1)
        ##for j in range(len(qi_trace)):
        ##    q = q_fcst(qi_trace[j], b_trace[j], di_trace[j], forecast_period, returned_months_trunc, dmin, shift_month_trace[j], shift_dt_trace[j], min_prod)
        ##    q_plt = np.append(prod_raw['GAS'][:prod_max_month], q)
        ##    q_trace = np.append(q_trace, q_plt[:600].sum())
        ##    plt.plot(returned_months, q_plt, sns.xkcd_rgb['denim blue'], alpha=0.005, lw=0.5, zorder=1)
        plt.scatter(prod_raw['MONTH'], prod_raw['GAS'], color='r', s=20, zorder=10)
        ##plt.scatter(prod_outliers['MONTH'], prod_outliers['GAS'], color='r', s=20, zorder=10)
        plt.xlim(0, 600)
        plt.ylim(min_prod, 500000)
        ax.set_yscale('log')
        plt.xlabel('Months on Production')
        plt.ylabel('Mcf')
        save_path = propnum + ' All Lines'
        plt.savefig(save_path)
        plt.close()

    p50 = np.percentile(q_trace, [49, 51])
    p50_range = np.where((q_trace >= p50[0]) & (q_trace <= p50[1]))
    forecasts[i, 0] = qi_trace[p50_range].mean()
    forecasts[i, 1] = b_trace[p50_range].mean()
    forecasts[i, 2] = di_trace[p50_range].mean()
    forecasts[i, 3] = shift_month_trace[p50_range].mean()
    forecasts[i, 4] = shift_dt_trace[p50_range].mean()
    start_shift.loc[propnum] = prod_max_month

    stop_time = time.time()
    elapsed_time = (stop_time - start_time) / 60
    print('run time: {:0.2f} minutes'.format(elapsed_time))

start_shift.to_csv('propnums.csv', sep=',')
np.savetxt('forecasts.csv', forecasts, fmt='%0.4f', delimiter=',', newline='\r\n')
    

