import os

if __name__ == '__main__':  # This is a temp file
    this_dir = '/mnt/e/Work/PhD/Articles/CIF_finder/development/structure_finder_2023-05-10T18-55-41-600146'
    n_cpu = 16

    os.system(f"python train_model.py {this_dir} -n {n_cpu} -s")
    os.system(f"python train_model.py {this_dir} -n {n_cpu}")