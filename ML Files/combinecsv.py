import pandas as pd
import glob

# find every csv inside year folders
files = glob.glob("optionscsv/**/*.csv", recursive=True)
print("Total CSVs:", len(files))   # expect 156

# read & merge
dfs = (pd.read_csv(f, sep=",", quotechar='"', skipinitialspace=True, low_memory=False) for f in files)
df  = pd.concat(dfs, ignore_index=True)

# save (choose one)
df.to_csv("spx_all_2010_2023.csv", index=False)      # big, plain text
# df.to_parquet("spx_all_2010_2023.parquet", index=False)  # compressed + faster

print(df.shape)
