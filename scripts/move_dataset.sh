#!/bin/bash

# Define the source and destination paths
source_folder="/home/xucao2/VLM_experiment/VCog/dataset/task1/tf3/pd"
destination_folder="/home/xucao2/VLM_experiment/VCog/testset/marsvqa"

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Loop through all subfolders in the source folder
for subfolder in "$source_folder"/*; do
  if [ -d "$subfolder" ]; then
    # Get the base names of the source and subfolder
    parent_folder_name=$(basename "$source_folder")
    subfolder_name=$(basename "$subfolder")
    
    # Construct the new subfolder name
    new_subfolder_name="${parent_folder_name}_${subfolder_name}"
    
    # Copy the subfolder to the destination with the new name
    cp -r "$subfolder" "$destination_folder/$new_subfolder_name"
  fi
done

echo "Subfolders have been renamed and copied to the destination folder."