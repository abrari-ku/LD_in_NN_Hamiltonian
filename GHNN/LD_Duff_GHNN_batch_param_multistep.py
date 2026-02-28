import os,sys, argparse
import math
import ghnn
import pandas as pd
import numpy as np
import h5py
import time
import datetime
import traceback
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

def distance_vector(u1, u2, p=1):
    
    """Calculate the p-norm distance between vectors u1 and u2, using only the first 4 elements
    
    Args:
        u1: First vector
        u2: Second vector
        p: Power for the p-norm calculation (default=1)
    
    Returns:
        d_a: Sum of the p-norm distances
    """
    d_a = np.sum(np.abs(u1 - u2) ** p)
    return d_a
    

def compute_ij_safe(args, my_net, tend, Np, p, xg, yg, dt, step_mult):
    """Worker function with error handling"""
    i, j = args
    try:
        x0 = np.array([xg[i], yg[j]])
        
        predict = my_net.predict_path(x0, tend)
        combined = np.vstack((predict["q_A"].values, predict["p_A"].values)).T
        
        step = int(step_mult)
        if step <= 0:
            raise ValueError("step_mult must be a positive integer")

        n_steps = Np // step
        ld_value = (step * dt) * sum(
            distance_vector(combined[k + step], combined[k], p)
            for k in range(0, n_steps * step, step)
        )
        return i, j, ld_value, None  # None indicates no error
    
    except Exception as e:
        warnings.warn(f"Failed at ({i},{j}): {str(e)}")
        return i, j, np.nan, traceback.format_exc()  # Return NaN + error info


def parallel_LD_computation(my_net, xg, yg, tend, Np, p, n_workers, dt, step_mult):
    nx, ny = len(xg), len(yg)
    LD = np.full((nx, ny), np.nan)  # Initialize with NaNs

    # Generate all (i,j) pairs
    indices = [(i, j) for i in range(nx) for j in range(ny)]

    # Create error log
    error_log = []
    
    with Pool(n_workers) as pool:
        results = pool.imap_unordered(
            partial(
                compute_ij_safe,
                my_net=my_net,
                tend=tend,
                Np=Np,
                p=p,
                xg=xg,
                yg=yg,
                dt=dt,
                step_mult=step_mult,
            ),
            indices
        )
        
        for i, j, val, error in results:
            if error is not None:
                error_log.append(f"Point ({i},{j}): {error}")
            LD[i, j] = val
    
    # Save results with error metadata
    return LD, error_log



def main():
    parser = argparse.ArgumentParser(description='Calculate LD for trained NNmodel ')
    parser.add_argument('--model', type=str, required=True, choices=['SympNet', 'HenonNet', 'GHNN'],
                      help='Model to calculate')
    parser.add_argument('--folder', type=str, 
                      default='duffing_infer_200_train_300725',
                      help='Path to saved nn model (default: duffing_infer_200_train_300725)')
    parser.add_argument('--n_workers', type=int, 
                  default=os.cpu_count(),  # Fallback to all cores if not set
                  help='Number of parallel workers (default: all cores)')
    parser.add_argument('--tend', type=float, 
                  default=10.0,
                  help='Integration time (default: 10.0)')
    parser.add_argument('--p', type=float, 
                  default=0.7,
                  help='Power parameter for distance calculation (default: 0.7)')
    parser.add_argument('--step_mult', type=int,
                  default=1,
                  help='Step multiplier for LD (e.g., 2 means use 2*dt steps)')
    parser.add_argument('--case', type=str,
                  default=None,
                  help='Case name (default: uses --folder value)')
     
    args = parser.parse_args()

        # Set case to folder if not explicitly provided
    if args.case is None:
        args.case = args.folder
    
    print(f"Using {args.n_workers} workers (SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')})")
    print(f"Parameters: model={args.model}, case={args.case}, tend={args.tend}, p={args.p}, step_mult={args.step_mult}")
    
  
    # Select the model as a list based on the task ID
    model_names = args.model

    # model_names = "GHNN"
    run_index = 1
    # case = f'duffing_{run}'
    case = args.case
    nn_paths = os.path.join('NeuralNets_GHNN', case, model_names, f'nn_{run_index}')
    my_net = ghnn.nets.net_from_dir(nn_paths)    
    nx = 400
    ny = 400
    # Add timing for better performance analysis
    print(f"Starting initial conditions processing at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # ic_start_time = time.time()

    xg = np.linspace(-1.5,1.5,nx)
    yg = np.linspace(-0.8,0.8,ny)
    LD = np.zeros([nx,ny])
    LDb = np.zeros([nx,ny])

    tend = args.tend
    dt = 0.1
    Np = int(tend/dt)
    p = args.p
    step_mult = args.step_mult

    output_file = os.path.join(nn_paths,'LD_output_duff_params.h5')
    group_name = f"LD_{model_names}_{case}_t_end_{tend}_gamma_{p}_dtm_{step_mult}"

    file_exists = os.path.exists(output_file)
        
    # Check if group already exists
    group_exists = False
    if file_exists:
        with h5py.File(output_file, 'r') as check_file:
            group_exists = group_name in check_file
    
    if group_exists:
        print(f"Group '{group_name}' already exists in the file. Skipping operation.")
        os._exit(0)  # Exit the script if group exists
    else:
        print(f"Group '{group_name}' does not exist. Continue operation...")
    
    print(f"Starting forward LD computation at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    forward_start = time.time()
    LD,err_log_forward = parallel_LD_computation(
            my_net=my_net,
            xg=xg,
            yg=yg,
            tend=tend,
            Np=Np,
            p=p,
            n_workers=args.n_workers,
                dt=dt,
                step_mult=step_mult
        ) 
    print(f"Forward LD computation completed in {time.time() - forward_start:.2f}s")

    print(f"Starting backward LD computation at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    backward_start = time.time()
    LDb,err_log_backward = parallel_LD_computation(
            my_net=my_net,
            xg=xg,
            yg=yg,
            tend=-tend,
            Np=Np,
            p=p,
            n_workers=args.n_workers,
                dt=dt,
                step_mult=step_mult
        ) 
    print(f"Backward LD computation completed in {time.time() - backward_start:.2f}s")
    with h5py.File(output_file, 'a') as f:
        g = f.create_group(group_name)
        g.create_dataset('LDf', data=LD)
        g.create_dataset('LDb', data=LDb)
        g.attrs.update({
            'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'xgrid': xg,
            'ygrid': yg,
            't_end': tend,
            'dt': dt,
            'step_mult': step_mult
        })
    print(f"Data written to {output_file} successfully.")
        
    print("\n=== Computation Successful ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.folder}")
    print(f"Initial conditions: {len(init_cond)} points")
    print(f"LD_forward range: [{np.nanmin(LD):.3f}, {np.nanmax(LD):.3f}]")
    print(f"LD_backward range: [{np.nanmin(LDb):.3f}, {np.nanmax(LDb):.3f}]")
    print(f"Forward errors: {len(err_log_forward)}")
    print(f"Backward errors: {len(err_log_backward)}")
    print(f"Output file: '{output_file}'")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runtime: {time.time() - start_time:.1f} seconds\n")
    
if __name__ == '__main__':
    start_time = time.time()
    main()
# for i in range(nx):

    # for j in range(ny):
        # x0 = np.array([xg[i],0,yg[j],0])
        # predict = my_net.predict_path(x0,-tend)
        # combined = np.vstack((predict["q_A"].values, predict["p_A"].values)).T
        # LDb[i,j] = sum([distance_vector(combined[k+1], combined[k], p) for k in range(Np)])
# LDb = parallel_compute_ldb(my_net, xg, yg, -tend, Np, p)

# with h5py.File('LD_output_forced_new.h5', 'a') as f:
#     g = f.create_group("LD2"+model_names+"_b_"+case+"_"+f'nn_{run_index}')
#     d = g.create_dataset('LD_f', data=LDb)
#     # d2 = g.create_dataset('LD_b', data=LDb)
#     g.attrs['Date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     g.attrs["xgrid"] = xg
#     g.attrs["ygrid"] = yg
#     g.attrs["t_end"] = -tend 