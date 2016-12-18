import os
import glob



input_path = "test-all"
output_path = "test-all-jpeg"

files = glob.glob(input_path+"/usr_*")

for file_i in files:
 output_file_name = file_i.split('.')[0].split(input_path+'/')[1]+'.jpeg'
 cmd = "convert " + file_i + " -resize 28x28\! " + output_path + "/" + output_file_name  
 os.popen(cmd).read()
 
