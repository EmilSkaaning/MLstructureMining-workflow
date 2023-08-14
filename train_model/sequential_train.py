import os

if __name__ == '__main__':  # This is a temp file
    this_dir = '/home/ekjaer//Projects/db/structure_finder_2023-07-17T21-24-00-901076'
    n_cpu = 120
    os.system(f"python train_model.py {this_dir} -n {n_cpu} -s")
    os.system(f"python train_model.py {this_dir} -n {n_cpu} -s -b")


