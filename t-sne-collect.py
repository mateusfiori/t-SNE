import os
import numpy as np

comments_file = open("./comments.csv", "w")
directory = os.fsencode("/Users/mateusfiori/Main Files/Mentoria/t-SNE/aclImdb/train/unsup")
prefix_path = "/Users/mateusfiori/Main Files/Mentoria/t-SNE/aclImdb/train/unsup"

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    
    if filename.endswith(".txt"): 
        data = open(os.path.join(prefix_path, filename), "r")
        comments_file.write(data.read().replace(",", "") + "\n")
        data.close()

comments_file.close()