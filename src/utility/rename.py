import os
import glob

# Specify the directory path and pattern
directory_path = r'D:\projects\RL-DeltaIoT\models'
file_pattern = '*.zip'  # Change the pattern to match the files you are interested in
names_dict = {'lr': 'LR', 'eps_min': 'EFE', 'batch_size': 'BS', 'gamma': 'DF'}
# Create the file path pattern by joining the directory path and file pattern
file_path_pattern = os.path.join(directory_path, file_pattern)

# Use glob to get a list of file paths matching the pattern
file_list = glob.glob(file_path_pattern)
new_file_list = []

# Print the list of files
print("List of Files:")
for file_path in file_list:
    new_file_path = file_path
    for key, value in names_dict.items():
        if key in file_path:
            new_file_path = new_file_path.replace(key, value)

    os.rename(file_path, new_file_path)

print('file_path')
