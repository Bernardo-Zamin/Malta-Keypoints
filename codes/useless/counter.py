import os 

folder_path = '/mnt/B-SSD/bernardo/output'
file_count = len(os.listdir(folder_path))

print(f"Number of files in the folder: {file_count}")
