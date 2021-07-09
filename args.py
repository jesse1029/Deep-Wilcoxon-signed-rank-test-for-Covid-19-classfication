import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type = str, default = '/ssd2/covid/data/')
    parser.add_argument("--train_batch", type = int, default = 10)
    parser.add_argument("--train_ct_batch", type = int, default = 16)
    parser.add_argument("--val_batch", type = int, default = 8)
    parser.add_argument("--epoch", type = int, default = 10)
    parser.add_argument("--lr", type = float, default = 0.0001)  # 0.00001  
    
    args = parser.parse_args([])
    
    return args