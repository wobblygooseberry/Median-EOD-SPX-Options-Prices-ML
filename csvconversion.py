import pandas as pd
import glob
import os

#paths
input_folder = "spx_eod_2023/"
output_folder = "optionscsv2023/"

os.makedirs("optionscsv2023", exist_ok=True)

#loop through all txt files in the input folder
for txt_file in glob.glob(input_folder + "*.txt"):
    #read txt file into a pandas dataframe
    df = pd.read_csv(txt_file, delimiter="\t")

    #create output file name with same base name but with .csv extension
    base_name = os.path.basename(txt_file).replace(".txt", ".csv")
    output_path = os.path.join(output_folder, base_name)

    #save the converted csv
    df.to_csv(output_path, index=False)
    print(f"Converted: {txt_file} to {output_path}")