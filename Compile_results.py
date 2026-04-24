import os
import argparse
import numpy as np

def main(experiment, alg):
    directories = next(os.walk('experiments'))[1]
    directories = [directory for directory in directories if "run" in directory]
    
    start_count = 0
    max_idx = 138
    
    pos_errors_sample = {i: [] for i in range(start_count, max_idx)}
    rot_errors_sample = {i: [] for i in range(start_count, max_idx)}
    vel_errors_sample = {i: [] for i in range(start_count, max_idx)}
    l_pos_errors_sample = {i: [] for i in range(start_count, max_idx)}
    u_pos_errors_sample = {i: [] for i in range(start_count, max_idx)}
    
    for directory in directories:
        files = next(os.walk('experiments/'+directory))[2]
        
        alg_files = [alg_file for alg_file in files if alg in alg_file and experiment in alg_file]
    
        for file in alg_files:
            print(directory + '/' + file)
            with open('experiments/' + directory + "/" + file, "r") as f:
                for line in f:
                    if "l_pos_error" in line:
                        e = line.find("l_pos_error")
                        idx = int(line[7:e])
                        s = line.find("= ")
                        val = float(line[s+2:])
                        # print(int(idx))
                        # print(float(val))
    
                        if idx in range(start_count, max_idx):
                            l_pos_errors_sample[idx].append(val)
                        
                    elif "u_pos_error" in line:
                        e = line.find("u_pos_error")
                        idx = int(line[7:e])
                        s = line.find("= ")
                        val = float(line[s+2:])
    
                        if idx in range(start_count, max_idx):
                            u_pos_errors_sample[idx].append(val)
                        
                    elif "pos_error" in line:
                        e = line.find("pos_error")
                        idx = int(line[7:e])
                        s = line.find("= ")
                        val = float(line[s+2:])
    
                        if idx in range(start_count, max_idx):
                            pos_errors_sample[idx].append(val)
                        
                    elif "rot_error" in line:
                        e = line.find("rot_error")
                        idx = int(line[7:e])
                        s = line.find("= ")
                        val = float(line[s+2:])
    
                        if idx in range(start_count, max_idx):
                            rot_errors_sample[idx].append(val)
                        
                    elif "vel_error" in line:
                        e = line.find("vel_error")
                        idx = int(line[7:e])
                        s = line.find("= ")
                        val = float(line[s+2:])
    
                        if idx in range(start_count, max_idx):
                            vel_errors_sample[idx].append(val)
    
    pos_error = 0
    rot_error = 0
    l_pos_error = 0
    u_pos_error = 0
    vel_error = 0
    
    for j in range(start_count, max_idx):
        pos_error += sum(pos_errors_sample[j]) / len(pos_errors_sample[j])
        rot_error += sum(rot_errors_sample[j]) / len(rot_errors_sample[j])
        l_pos_error += sum(l_pos_errors_sample[j]) / len(l_pos_errors_sample[j])
        u_pos_error += sum(u_pos_errors_sample[j]) / len(u_pos_errors_sample[j])
        vel_error += sum(vel_errors_sample[j]) / len(vel_errors_sample[j])
    
    print('rot_error = ' + str(rot_error/(max_idx-start_count)))
    print('pos_error = ' + str(pos_error/(max_idx-start_count)))
    print('u_pos_error = ' + str(u_pos_error/(max_idx-start_count)))
    print('l_pos_error = ' + str(l_pos_error/(max_idx-start_count)))
    print('vel_error = ' + str(vel_error/(max_idx-start_count)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result compilation after running an experiment through test.py")
    
    parser.add_argument("-a", "--algorithm", default="pigdm", help="Choose algorithm, either pigdm/cfg")
    
    parser.add_argument("-e", "--experiment", default="default", help="Choose experiment to be compiled")
    
    args = parser.parse_args()
    main(args.experiment,args.algorithm)