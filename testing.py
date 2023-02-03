from datetime import datetime, timedelta

inputa = '2023-10'
dts_split = inputa.split('-')
dtn_split = datetime.now().strftime('%Y-%m').split('-')
new_start_year = (int(dtn_split[0]) - int(dts_split[0])) + 1
print(new_start_year)
