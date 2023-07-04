import pandas as pd
import numpy as np
from collections import Counter

fields = ['Timestamps', 'Path Clearance', 'Avg Goal Distance']

data1 = pd.read_csv('text_result.csv').set_index('Map')
data2 = pd.read_csv('result.csv').set_index('Map')

combined = data1.join(data2, how='outer', lsuffix='_ours', rsuffix='_baseline')

analysis = {}

for index, row in combined.iterrows():
    map_result = {}
    winners = []
    for field in fields:
        v1 = row[f'{field}_ours']
        v2 = row[f'{field}_baseline']

        less_is_better = (field in ['Timestamps','Avg Goal Distance'])
        if pd.isnull(v1):
            winner, pct_diff = 'baseline', np.nan
        elif pd.isnull(v2):
            winner, pct_diff = 'ours', np.nan
        else:
            if less_is_better:
                winner = 'ours' if v1 <= v2 else 'baseline'
            else:
                winner = 'ours' if v1 >= v2 else 'baseline'
            # print(v1,v2)
            pct_diff = abs((v1-v2)/max(v1,v2)) * 100
            # print(pct_diff,'%')
        winners.append(winner)
        map_result[field+' Winner'] = winner
        map_result[field+' Percent'] = pct_diff
       
    # Count the winners and assign the overall victory
    counter = Counter(winners)
    map_result['Overall Winner'] = max(counter, key=counter.get)      

    analysis[index] = map_result 

analysis_df = pd.DataFrame(analysis).T
print(analysis_df)

analysis_df.to_csv('result_compare.csv')