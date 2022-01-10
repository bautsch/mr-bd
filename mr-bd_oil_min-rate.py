import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as t
import theano as th
import math
import time
import datetime as dt
import smtplib
pd.options.mode.chained_assignment = None

file_path = r''
start_date = dt.datetime(2017, 2, 28)
max_run_yrs = 50
major = 'oil'
if major == 'oil':
    c = 'g'
    ylabel = 'Bbl'
if major == 'gas':
    c = 'r'
    ylabel = 'Mcf'
mcmc_trials = 30000
burn = 10000
forecast_period = np.arange(0, 600, 1)
returned_months = np.arange(0, 600, 1)
dmin = 0.08
min_prod = 5 #bbls/month
sampling_trials = 1000
well_aggregation = 8
msg_count = 0

dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
print('Building data frame...')
data = pd.read_csv(file_path, delimiter=',', header=0, parse_dates=[1], date_parser=dateparse, dtype={major: int})
end_date = dt.datetime(start_date.year+max_run_yrs, start_date.month, start_date.day)
d_list = []
propnum_list = []

print('Extending data frame to max life...')
for _idx, propnum in enumerate(data.propnum.unique()):
##    if (_idx + 1) % 10 == 0:
##        print(_idx + 1, 'of', len(data.propnum.unique()))
    d = data[data.propnum == propnum]['p_date'].max()
    d_temp = []
    while d < end_date:
        d = (dt.timedelta(days=4) + d).replace(day=28) + dt.timedelta(days=4)
        d = d - dt.timedelta(days=d.day)
        d_temp.append(d)
    propnum_temp = [propnum]*len(d_temp)
    propnum_list.extend(propnum_temp)
    d_list.extend(d_temp)

temp_df = pd.DataFrame({'propnum': propnum_list, 'p_date': d_list})
data = data.append(temp_df, ignore_index=True)

print('Calculating months on production...')
for _idx, propnum in enumerate(data.propnum.unique()):
    if _idx == int(len(data.propnum.unique())/2):
        print('Thinking really hard...')
    d1 = (data[data.propnum == propnum]['p_date'].dt.year - data[data.propnum == propnum]['p_date'].min().year)*12
    d2 = data[data.propnum == propnum]['p_date'].dt.month - data[data.propnum == propnum]['p_date'].min().month
    data.loc[data.propnum == propnum, 'month'] = d1 + d2

data['month'] = data['month'].astype(int)
data.sort_values(['propnum', 'p_date'], inplace=True)

print('Data frame construction complete.')
print('Starting up inferential engine...')
num_wells = len(data.propnum.unique())
propnums = pd.Series(index=data.propnum.unique())
forecasts = np.empty((num_wells, 3))
all_qi_traces = pd.Series(np.nan, index=range(0, (mcmc_trials-burn)*num_wells))
all_b_traces = pd.Series(np.nan, index=range(0, (mcmc_trials-burn)*num_wells))
all_di_traces = pd.Series(np.nan, index=range(0, (mcmc_trials-burn)*num_wells))
all_q_traces = pd.Series(np.nan, index=range(0, (mcmc_trials-burn)*num_wells))


def q_fcst_t(qi, b, di, forecast_period, returned_months, dmin, min_prod):
    q = qi / np.power(1+b*di*forecast_period, 1/b)
    d = t.concatenate([t.arange(1, 2), t.extra_ops.diff(q)*-1 / q[:-1]])
    terminal_mask = d < dmin/12
    terminal_start = t.argmax(terminal_mask)
    terminal_period = t.arange(terminal_start, len(forecast_period), 1) - (terminal_start - 1)
    qi_exp = q[terminal_start-1]
    q_exp = qi_exp*np.exp(-dmin/12*terminal_period)
    q_arps = q[:terminal_start]
    forecast = t.concatenate([q_arps, q_exp])
    min_prod_mask = forecast < min_prod
    min_prod_start = t.argmax(min_prod_mask)
    post_min_period = t.arange(min_prod_start, len(forecast_period), 1) - (min_prod_start - 1)
    pre_min = forecast[:min_prod_start]
    post_min = 0*post_min_period
    forecast = t.concatenate([pre_min, post_min])
    return forecast[returned_months]

def q_fcst(qi, b, di, forecast_period, returned_months, dmin, min_prod):
    q = qi / np.power(1+b*di*forecast_period, 1/b)
    d = np.concatenate((np.arange(1, 2), np.divide(np.diff(q)*-1, q[:-1])))
    terminal_mask = d < dmin/12
    terminal_start = terminal_mask.argmax()
    terminal_period = np.arange(terminal_start, len(forecast_period), 1) - (terminal_start - 1)
    qi_exp = q[terminal_start-1]
    q_exp = qi_exp*np.exp(-dmin/12*terminal_period)
    q_arps = q[:terminal_start]
    forecast = np.concatenate([q_arps, q_exp])
    min_prod_mask = forecast < min_prod
    min_prod_start = min_prod_mask.argmax()
    post_min_period = len(forecast_period) - min_prod_start
    pre_min = forecast[:min_prod_start]
    post_min = np.zeros(post_min_period)
    forecast = np.concatenate([pre_min, post_min])
    return forecast[returned_months]

def loss_func(well_prod, qi, b, di, forecast_period, dmin, start_shift):
    q_est = q_fcst(qi, b, di, forecast_period, well_prod['month'] >= start_shift, dmin)
    err1 = (well_prod[major] - q_est).sum()
    err2 = abs(well_prod[major] - q_est).sum()
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

for i, propnum in enumerate(data.propnum.unique()):
    start_time = time.time()
    prod_raw = data[data.propnum == propnum]
    prod_max = prod_raw[major].max()
    prod_max_month = prod_raw[prod_raw[major] == prod_max]['month'].values[0]
    ##rolling_median = prod_raw[major].rolling(window=5, center=True).median()
    ##diff = abs(prod_raw[major] - rolling_median)
    ##outliers = (diff / rolling_median > 50) & (prod_raw['month'] > 10)
    ##prod_filtered = prod_raw[~outliers]
    ##prod_outliers = prod_raw[outliers]
    prod_filtered = prod_raw

    if prod_max_month > 6:
        prod_max_month = 1

    if prod_max_month > 0:
        prod_filtered = prod_filtered[prod_filtered['month'] >= prod_max_month]
        prod_filtered.loc[:, ['month']] = prod_filtered.loc[:, ['month']] - prod_max_month

    prod_filtered = prod_filtered[prod_filtered.loc[:,major] !=0]
    prod_filtered = prod_filtered[prod_filtered[major].notnull()]

    print(i, 'of', len(data.propnum.unique()), propnum)
    ##print('max diff:', (diff / diff.std()).max())
    ##print('num outliers:', outliers.sum())
    print('max prod month:', prod_max_month)
    print('forecast months:', prod_filtered.shape[0])
    
    with pm.Model() as Arps:

        if prod_filtered.shape[0] == 1:
            qi = pm.Uniform('qi', 0, 25000)
            b = pm.Uniform('b', lower=0.8, upper=1.5)
            di = pm.Uniform('di', lower=0.01, upper=0.6)
        else:
            qi = pm.Uniform('qi', 0, 25000)
            b = pm.Uniform('b', lower=0.01, upper=2.0)
            di = pm.Uniform('di', lower=0.01, upper=1)
        q = q_fcst_t(qi, b, di, forecast_period, prod_filtered['month'].values, dmin, min_prod)
        sd = pm.HalfNormal('sd', sd=0.1)

        obs = pm.StudentT('obs', nu=1, mu=q, lam=sd, observed=prod_filtered[major].values)

        print('Generating traces...')
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
        q_trace = np.empty(len(qi_trace))
        if np.random.rand() < 0.05:
            print('Making up numbers...')
        else:
            print('Quantifying reserves...')
        for j in range(len(qi_trace)):
            all_qi_traces[(i+1)*j] = qi_trace[j]
            all_b_traces[(i+1)*j] = b_trace[j]
            all_di_traces[(i+1)*j] = di_trace[j]
            q_trace[j] = q_fcst(qi_trace[j], b_trace[j], di_trace[j], forecast_period, returned_months_trunc, dmin, min_prod).sum()
            all_q_traces[(i+1)*j] = q_trace[j].sum()

        p10 = np.percentile(q_trace, [9, 11])
        p10_range = np.where((q_trace >= p10[0]) & (q_trace <= p10[1]))
        qi_p10 = qi_trace[p10_range].mean()
        b_p10 = b_trace[p10_range].mean()
        di_p10 = di_trace[p10_range].mean()
        q_p10 = q_fcst(qi_p10, b_p10, di_p10, forecast_period, returned_months_trunc, dmin, min_prod)
        
        p50 = np.percentile(q_trace, [49, 51])
        p50_range = np.where((q_trace >= p50[0]) & (q_trace <= p50[1]))
        qi_p50 = qi_trace[p50_range].mean()
        b_p50 = b_trace[p50_range].mean()
        di_p50 = di_trace[p50_range].mean()
        q_p50 = q_fcst(qi_p50, b_p50, di_p50, forecast_period, returned_months_trunc, dmin, min_prod)

        p90 = np.percentile(q_trace, [89, 91])
        p90_range = np.where((q_trace >= p90[0]) & (q_trace <= p90[1]))
        qi_p90 = qi_trace[p90_range].mean()
        b_p90 = b_trace[p90_range].mean()
        di_p90 = di_trace[p90_range].mean()
        q_p90 = q_fcst(qi_p90, b_p90, di_p90, forecast_period, returned_months_trunc, dmin, min_prod)
            
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
        plt.plot(forecast_period[prod_max_month:], q_p50, sns.xkcd_rgb['denim blue'], alpha=0.75, lw=1, zorder=1, label='P50 EUR: {:,d}'.format(int(q_p50.sum())))
        plt.scatter(prod_raw['month'], prod_raw[major], color=c, s=20, zorder=10)
        plt.xlim(0, 600)
        plt.ylim(min_prod, 50000)
        ax.set_yscale('log')
        plt.xlabel('Months on Production')
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', fancybox=True, framealpha=0.5)
        save_path = propnum + ' P50 Forecast'
        plt.savefig(save_path)
        plt.close()
        print('P50 EUR: {:,d}'.format(int(q_p50.sum()))) 

    print('Writing output to data frame...')
    forecasts[i, 0] = qi_p50
    forecasts[i, 1] = b_p50
    forecasts[i, 2] = di_p50
    propnums.loc[propnum] = prod_max_month
    data.loc[(data[major].notnull()) & (data.propnum == propnum), 'forecast'] = data[major]
    data.loc[(data[major].isnull()) & (data.propnum == propnum), 'forecast']
    fcst_start = data.loc[data[major].notnull() & (data.propnum == propnum), 'month'].max() - prod_max_month
    data.loc[data[major].isnull() & (data.propnum == propnum) & (data.month < q_p50.shape[0] + prod_max_month + 1), 'forecast'] = q_p50[fcst_start:]
    
    stop_time = time.time()
    elapsed_time = (stop_time - start_time) / 60
    if elapsed_time > 3 and msg_count < 5:
        msg_count += 1
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login('', '')
        s.sendmail('', '', 'Slow: ' + str(i))
        s.close()
    print('Run time: {:0.2f} minutes'.format(elapsed_time))

propnums.to_csv('propnums.csv', sep=',')
np.savetxt('forecasts.csv', forecasts, fmt='%0.4f', delimiter=',', newline='\r\n')

avg_p50_fcst = data.groupby(['month'])['forecast'].mean().sum()
print('Average P50 EUR: {:,d}'.format(int(avg_p50_fcst)))
print('Finished.')

