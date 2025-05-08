import os
import sys
import pandas as pd

if not os.path.isdir("./data"):
    print("Scripts expects data in /data directory")
    sys.exit()

if not os.path.isdir("./data/sont_a"):
    print("Scripts expects sont_a/ual-1 log files in directory /data/sont_a")
    sys.exit()

files = os.listdir("./data/sont_a")
data = list()
for file in files:
    p = pd.read_csv("./data/sont_a/" + file)
    data.append(p)

df = pd.concat(data)
df.set_index("timestamp_hr", inplace=True)
df.index.rename("datetime", inplace=True)
df = df.loc[:, ['CO',
                'NO',
                'NO2',
                'O3',
                'pm25',
                'pm10',
                'RAW_ADC_CO_W',
                'RAW_ADC_CO_A',
                'RAW_ADC_NO_W',
                'RAW_ADC_NO_A',
                'RAW_ADC_NO2_W',
                'RAW_ADC_NO2_A',
                'RAW_ADC_O3_W',
                'RAW_ADC_O3_A',
                'sht_humid',
                'sht_temp']]

df.to_csv("./data/minute_data_ual-1.csv")

print(df)
print(df.columns)


