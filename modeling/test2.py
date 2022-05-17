import clean_parallel_init as t
import pandas as pd
import multiprocessing as mp
import time
from modeling.clean_parallel import cp
cp = cp()
cpt = t.cpt()

df = pd.read_csv('../trial_data/AAPL.csv')
data = pd.DataFrame({0:df['content']})


init_time_list = []
for i in [250,500,750,1000,1250,1500,1750,2000]:
    start = time.time()
    pool = mp.Pool(processes=mp.cpu_count(), initializer=t.work, initargs=(t.data[0][0:i],))
    result = pool.map(t.clean_mlm_parallel_init, range(0,len(t.data[0][0:i])))
    pool.close()
    init_time_list.append(time.time()-start)

regular_time_list = []
for i in [250,500,750,1000,1250,1500,1750,2000]:
    start = time.time()
    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.map(cp.clean_mlm_parallel, [article for article in t.data[0][0:i]])
    pool.close()
    regular_time_list.append(time.time()-start)

import matplotlib.pyplot as plt

x = [250,500,750,1000,1250,1500,1750,2000]
plt.plot(x, init_time_list,label="Init",color='red')
plt.plot(x, regular_time_list,label="Reg",color='blue')
plt.legend()
plt.show()


for i in [2,4,8,16,32,64,128,256]:
    start = time.time()
    pool = mp.Pool(processes=mp.cpu_count(), initializer=t.work, initargs=(t.data[0][0:1000],))
    result = pool.map(t.clean_mlm_parallel_init, range(0,len(t.data[0][0:1000])),i)
    pool.close()
    print(f"Time Chunk {i}",time.time()-start)


for i in [2,4,8,16,32,64,128,256]:
    start = time.time()
    pool = mp.Pool(processes=mp.cpu_count(), initializer=t.work, initargs=(t.data[0][0:1000],))
    result = pool.imap(t.clean_mlm_parallel_init, range(0,len(t.data[0][0:1000])),i)
    pool.close()
    print(f"Time Chunk imap {i}",time.time()-start)



pool = mp.Pool(processes=mp.cpu_count(), initializer=t.work, initargs=(data[0][0:1000],))
values = pool.imap(cpt.clean_mlm_parallel_init, range(0,len(data[0][0:1000])))
result = []
for i in values:
    result.append(i)
pool.close()