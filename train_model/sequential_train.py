import os

if __name__ == '__main__':  # This is a temp file
    this_dir = '/home/ekjaer/Projects/db/ciff_090_test'
    n_cpu = 120
    n_datas = [200, 150, 100, 50]
    for n_data in n_datas:
        print(n_data)
        os.system(f"python train_model.py {this_dir} -n {n_cpu} -s -d {n_data}")
    
    #os.system(f"python train_model.py {this_dir} -n {n_cpu}")
