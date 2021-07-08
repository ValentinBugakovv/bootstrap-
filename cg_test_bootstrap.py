import os
import argparse
import numpy as np
import pandas as pd
import vertica_python
import time

from scipy.stats import norm
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')


CONN_INFO = {'host': '192.168.246.202',
             'port': 5433,
             'user': '',
             'password': '',
             'database': 'delimobil_dwh'
            }

# Reading the text of the SQL query from the txt file
def create_query(query_path):
    with open(query_path, 'r') as f:
        query_text = f.read()
    return query_text

# Loading data
def get_data_from_base(query, CONN_INFO):
    with vertica_python.connect(**CONN_INFO) as connection:
        t1 = time.time()
        cur = connection.cursor()
        query = cur.execute(query)
        columnList = [d.name for d in cur.description]
        data = pd.DataFrame(cur.fetchall(), columns=columnList, dtype = np.float64)
        t2 = time.time()
    print('Data loaded, {:.5} sec.'.format(t2-t1))
    return data

def execute_query(query, CONN_INFO):
    with vertica_python.connect(**CONN_INFO) as connection:
        t1 = time.time()
        cur = connection.cursor()
        cur.execute(query) 
        t2 = time.time()
    print('Query executed, {:.5} sec.'.format(t2-t1))

def get_bootstrap(
    data_column_1, # values of the first sample
    data_column_2, # values of the second sample
    boot_it = 1000, # number of bootstrap subsamples
    statistic = np.mean, # statistic
    bootstrap_conf_level = [0.80, 0.95, 0.99], # significance level
    boot_len = None
):
    if boot_len == None:
#         boot_len = max(len(data_column_1),len(data_column_2))
#         boot_len = int(0.3*(len(data_column_1) + len(data_column_2))) # 30% from the total number of users
        # not less than the number of elements in the minimum sample
        boot_len = max(int(0.3*(len(data_column_1) + len(data_column_2))), 
                       min(len(data_column_1),len(data_column_2)))
    print('boot_len = ',boot_len)
    print('boot_it = ',boot_it)
    boot_data = []
    for i in tqdm(range(boot_it)): # extracting subsamples
        samples_1 = data_column_1.sample(
            boot_len, 
            replace = True 
        ).values
        
        samples_2 = data_column_2.sample(
            boot_len, # to preserve the variance, we take the same sample size
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1-samples_2)) 
    pd_boot_data = pd.DataFrame(boot_data)
    # quantiles    
    alphas = []
    for conf_level in bootstrap_conf_level:
        alphas.append((1 - conf_level)/2)
        alphas.append(1 - (1 - conf_level) / 2)
    quants = pd_boot_data.quantile(alphas).sort_index()
    # part of the histogram < 0    
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    # part of the histogram > 0
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}

def significance(p_value, a):
    if p_value > 1-a:
        return False
    else:
        return True

def boot_res_df(data, cg_list):
    metric_cols = {'Delta_charges_bill_per_client':'sum_bill',
    'Delta_charges_bill_per_riders':'sum_bill',
    'Delta_charges_cost_per_client':'sum_cost',
    'Delta_charges_cost_per_riders':'sum_cost',
    'Delta_share_of_riders':'was_ride',
    'Delta_TA_number_per_client':'cnt_rds',
    'Delta_TA_number_per_riders':'cnt_rds'              
    }
    boot_cols = ['id','name',
            'period_start','period_end','period_start_month',
            'TG_clients','CG_clients',
            'TG_riders','CG_riders',
            'bootstrap_samples_len',
            'Delta_name','Delta_value',
            'Delta_value_bootstrap','p_value',
            'is_significant_80','is_significant_95','is_significant_99'
                ]
    boot_results = pd.DataFrame(columns = boot_cols)
    for cg_name in cg_list:

        # campain id
        cg_ids = data.loc[data['name'] == cg_name, 'id'].unique()
        assert len(cg_ids) == 1, "Several ids for 1 cg name"
        cg_id = cg_ids[0]
        # period_start
        cg_period_starts = data.loc[data['name'] == cg_name, 'period_start'].unique()
        assert len(cg_period_starts) == 1, "Several period_start for 1 cg name"
        cg_period_start = cg_period_starts[0]
        # period_end
        cg_period_ends = data.loc[data['name'] == cg_name, 'period_end'].unique()
        assert len(cg_period_ends) == 1, "Several period_end for 1 cg name"
        cg_period_end = cg_period_ends[0]
        # period_start_month
        cg_period_start_months = data.loc[data['name'] == cg_name, 'period_start_month'].unique()
        assert len(cg_period_start_months) == 1, "Several period_start_month for 1 cg name"
        cg_period_start_month = cg_period_start_months[0]

        cols = ['user_id','is_cg','sum_bill','sum_cost','cnt_rds']
        cg_data = data.loc[data['name'] == cg_name, cols]
        cg_data['was_ride'] = np.where(cg_data['cnt_rds']>0, 1, 0)
        print('{}: TG = {}, CG = {}'.format(cg_name, 
                                            cg_data[cg_data['is_cg'] == False].shape[0],
                                            cg_data[cg_data['is_cg'] == True].shape[0]))
        # bootstrap
        for delta_name in metric_cols:
            col = metric_cols[delta_name]
            if (delta_name == 'Delta_share_of_riders') or ('client' in delta_name):
                cg = cg_data.loc[(cg_data['is_cg'] == True), col].fillna(0)
                tg = cg_data.loc[(cg_data['is_cg'] == False), col].fillna(0) 
            else:
                cg = cg_data.loc[(cg_data['is_cg'] == True)&(cg_data['was_ride']>0), col]
                tg = cg_data.loc[(cg_data['is_cg'] == False)&(cg_data['was_ride']>0), col]
            tg_size = cg_data[cg_data['is_cg'] == False].shape[0]
            cg_size = cg_data[cg_data['is_cg'] == True].shape[0]
            tg_riders_size = cg_data[(cg_data['is_cg'] == False)&(cg_data['was_ride']>0)].shape[0]
            cg_riders_size = cg_data[(cg_data['is_cg'] == True)&(cg_data['was_ride']>0)].shape[0]
            print(delta_name,col)
            delta_val = tg.mean()-cg.mean()
            bootdata = get_bootstrap(tg, cg, 
                            boot_it = 10000, # number of bootstrap subsamples
                            statistic = np.mean, # statistic
                            bootstrap_conf_level = [0.80,0.95,0.99], # significance level
                            boot_len = None)
            bootstrap_samples_len = int(0.3*(len(cg) + len(tg)))
            p_value = bootdata['p_value']
            is_significant_80 = significance(p_value, 0.8)
            is_significant_95 = significance(p_value, 0.95)
            is_significant_99 = significance(p_value, 0.99)
            boot_results.loc[len(boot_results)] = [cg_id,cg_name,
                                                   cg_period_start,cg_period_end,cg_period_start_month,
                                                   tg_size,
                                                   cg_size,
                                                   tg_riders_size,
                                                   cg_riders_size,
                                                   bootstrap_samples_len,
                                                   delta_name,
                                                   delta_val,
                                                   np.mean(bootdata['boot_data']),
                                                   p_value,
                                                   is_significant_80,
                                                   is_significant_95,
                                                   is_significant_99
                                                   ]
    boot_results['id'] = boot_results['id'].astype(int)        
    return boot_results


parser = argparse.ArgumentParser()
parser.add_argument('--campaings', type=str, help='value could be new or all', default='new',
                       dest='campaings')
parser.add_argument('--table_name', type=str, help='name of table', default='public.kb_cg_bootstrap',
                       dest='table_name')
    
def main():
    args = parser.parse_args()
    campaings = args.campaings
    table_name = args.table_name
    print(campaings, table_name)
    # get list of all campaings
    q2 = f"select * from public.ef_tmp_res_v1 \
            where camp_w_CG = 1 \
            ;"
    data = get_data_from_base(q2, CONN_INFO)
    all_cg_list = data.name.unique()
    if campaings == 'all':
        t1 = time.time()
        boot_results = boot_res_df(data, all_cg_list)
        t2 = time.time()
        print('Time to get bootstrap results for all campaings list: {} s'.format(t2-t1))
    if campaings == 'new':
        # # get list of already processed campaings
        q3 = "select * from {};".format(table_name) # public.kb_cg_bootstrap
        processed_data = get_data_from_base(q3, CONN_INFO)
        processed_campaigns = processed_data['name'].unique()
        new_cg_list = list(set(all_cg_list) - set(processed_campaigns))
        print(new_cg_list)
        t1 = time.time()
        boot_results = boot_res_df(data, new_cg_list)
        t2 = time.time()
        print('Time to get bootstrap results for new campaings list: {} s'.format(t2-t1))
    dt = str(datetime.datetime.now())[:16].replace(' ','').replace('-','').replace(':','')    
    path = './data/output/boot_results'+ dt +'.csv'
    boot_results.to_csv(path, encoding='utf-8',
                        index = None, sep = ';')

    
if __name__ == '__main__':
    main()