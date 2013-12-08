'''
Created on Nov 16, 2012

@author: jared.oyler
'''
import os

def download(path_fftp,out_path):
    
    os.chdir(out_path)
    fftp = open(path_fftp)
    for line in fftp.readlines():
        
        line = line.strip()
        if line[-3:] == "hdf":
            os.system("".join(["wget ",line]))

if __name__ == '__main__':
    download("/projects/daymet2/climate_office/modis/MYD11A2_dwnld.txt", 
             "/projects/daymet2/climate_office/modis/MYD11A2.005")
    
#    download("/projects/daymet2/climate_office/modis/data_url_script_2012-11-16_180230.txt", 
#             "/projects/daymet2/climate_office/modis/MOD13Q1.005")