# ==============================================================================
# Echo State Network (ESN) Training for Duffing Oscillator with Time Reversal
# ==============================================================================
# This script trains a reservoir computing network (ESN) to learn the dynamics
# of a Duffing oscillator system. It optimizes ESN hyperparameters using CMA-ES
# and trains both forward and backward (time-reversed) output matrices.
#
# Key components:
# - CMA-ES optimization for hyperparameters (spectral radius, leaky rate, etc.)
# - Bidirectional training (forward and backward in time)
# - Multiple training trajectories handling
# - HDF5 data I/O for datasets and trained models
# ==============================================================================

using Pkg
using Parameters,Random,CMAEvolutionStrategy,HDF5,Adapt, SparseArrays
using LoggingExtras, Dates, Printf
using ReservoirComputing

# Setup logging to file with timestamps
const date_format = "yyyy-mm-dd HH:MM:SS"
logger = FormatLogger("out2.log"; append=true) do io, args
    println(io, args._module, " | ", "[", args.level,": ", Dates.format(now(), date_format), "] ", args.message)
end;
global_logger(logger)
"""
    read_data(folder_name, data_file)

Load training and test data from HDF5 file for ESN training.

Returns:
- train_data: Input features for training (forward direction)
- labels_train: Training labels after washout removal (forward)
- test_data: Validation dataset
- train_rev: Time-reversed training features
- labels_train_rev: Time-reversed training labels
- N_run: Number of training trajectories
- n_valid: Number of validation trajectories
"""
function read_data(folder_name, data_file)
    train_test_file = h5open(folder_name*data_file,"r")
    
    # Load features and labels from HDF5 file
    train_test_data = read(train_test_file["features"],"block0_values")
    N_run = Int(size(train_test_data,2)/N_el_run)  # Number of training runs
    train_data = read(train_test_file["features"],"block0_values")
    labels = read(train_test_file["labels"],"block0_values") 
    # Prepare arrays for forward and backward training labels
    labels_train = Array{Float64, 2}(undef,2, N_run*(N_el_run-wash))
    labels_train_rev = Array{Float64, 2}(undef,2, N_run*(N_el_run-wash))
    
    # Create time-reversed versions for bidirectional training
    labels_rev = reverse(train_test_data,dims = 2)  # Reverse features in time
    train_rev = reverse(labels,dims = 2)            # Reverse labels in time
    
    # Remove washout period from labels for each training run
    for i in 1:N_run
        # Forward labels (excluding washout period)
        labels_train[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)]= labels[:, (i-1)*N_el_run+wash+1:i*N_el_run]
        # Backward labels (excluding washout period)
        labels_train_rev[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = labels_rev[:, (i-1)*N_el_run+wash+1:i*N_el_run]
    end
    # Load validation/test data
    test_set = read(train_test_file["test_features"],"block0_values")
    n_valid = Int(size(test_set,2)/N_el_run)  # Number of validation trajectories
    test_data=Array{AbstractArray{Float64,2},1}(undef,n_valid)
    
    # Extract each validation trajectory (includes washout + validation portion)
    for i in 1:n_valid
        test_data[i] = test_set[:,(i-1)*N_el_run+1:(i-1)*N_el_run+wash+length_valid] 
    end
    
    close(train_test_file)
    return train_data, labels_train, test_data, train_rev, labels_train_rev, N_run, n_valid
end

"""
    Loss(esn, Wout, last_state, validation_set)

Compute validation loss for ESN in generative mode.

Calculates the sum of L2 norms of prediction errors across all time steps.
"""
function Loss(esn,Wout,last_state, validation_set)
    N = size(validation_set,2)  # Number of time steps to predict
    # Generate predictions autonomously from the last state
    output = esn(Generative(N),Wout, last_state = last_state)
    err = output - validation_set
    # Return sum of L2 norms for each time step
    return sum(sqrt.(sum(abs2,err,dims = 1)))
end

"""
    obj(θ)

Objective function for CMA-ES optimization.

Computes average validation loss across all validation trajectories for given
ESN hyperparameters θ = [spectral_radius, leaky_coefficient, input_scaling, log(ridge_coefficient)].

Returns: Mean normalized validation error
"""
function obj(θ)
    # Build ESN with current hyperparameters and train on all runs
    esn, Wout = compute_esn_all_runs(θ)
    
    # Evaluate performance on each validation trajectory
    loss = zeros(n_valid);
    for i in 1:n_valid
        # Compute reservoir states for washout period
        states_test = create_states(esn,test_data[i][:,1:wash+1], washout = wash);
        # Compute loss on validation portion
        loss[i] = Loss(esn,Wout,states_test[:,1], test_data[i][:,wash+1:end]);
    end
    
    # Return average normalized loss
    return sum(loss)/(n_valid*length_valid)
end

"""
    compute_esn_all_runs(θ)

Create and train an ESN with given hyperparameters on all training runs.

Args:
- θ: Transformed hyperparameter vector [SR, α, σ, log(β)]

Returns:
- esn: Configured Echo State Network
- Wout: Trained output weight matrix
"""
function compute_esn_all_runs(θ)
    # Convert from optimization space [0,10] to physical parameter space
    θ = untransform(θ)
    SR, α, σ, logβ = θ;  # Spectral Radius, leaky coefficient, input scaling, log(ridge coeff)
    
    # Initialize ESN with first training run
    run_train_data = train_data[:, 1:N_el_run]
    esn = ESN(run_train_data,2, NW;  # 2 input dimensions, NW reservoir nodes
        reservoir = rand_sparse(; radius = SR, sparsity = sparsity),
        input_layer = weighted_init(;scaling = σ),
        reservoir_driver = RNN(leaky_coefficient = α),
        nla_type = NLADefault(),washout = wash,rng = Random.seed!(rnd_seed));

    
    # Store reservoir states from first run (after washout)
    all_train_states[:,1:(N_el_run-wash)] = esn.states

    # Collect reservoir states from all other training runs
    for i in 2:N_run
        filtered_block0_values = train_data[:, (i-1)*N_el_run+1:i*N_el_run]
        all_train_states[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = create_states(esn, filtered_block0_values, washout = wash)
    end

    # Train output weights using ridge regression on all collected states
    Wout = train(StandardRidge(exp(logβ)), all_train_states, labels_train)

    return esn, Wout
end

"""
    untransform(θ)

Map parameters from optimization space [0,10] to physical parameter space [lb, ub].
Used during optimization to convert CMA-ES parameters to actual ESN hyperparameters.
"""
function untransform(θ::AbstractArray{Float64, 1})
    return lb + (ub - lb).*θ/10
end

"""
    transform(θ)

Map parameters from physical space [lb, ub] to optimization space [0,10].
Used to initialize the optimization with physical parameter values.
"""
function transform(θ::Array{Float64, 1})
    return Adapt.adapt(typeof(θ), 10*(θ - lb)./(ub - lb))
end

"""
    fun_save_ESN(esn_name)

Save trained ESN model to HDF5 file.

Saves:
- ESN architecture and hyperparameters as attributes
- Forward and backward output weight matrices (Wout, Wout_rev)
- Reservoir matrix (sparse format if applicable)
- Input matrix (sparse format if applicable)
"""
function fun_save_ESN(esn_name)
    save_ESN = h5open("ESN_data_test2.h5","cw")
    g = create_group(save_ESN, esn_name)
    
    # Save ESN metadata
    attributes(g)["res_size"] = esn.res_size
    attributes(g)["res_driver"] = string(esn.reservoir_driver)
    res_d_str = (string(esn.reservoir_driver))
    attributes(g)["nla_type"] = string(esn.nla_type)
    attributes(g)["states_type"] = string(esn.states_type)
    
    # Save output weight matrices (forward and backward)
    write_dataset(g,"Wout",Wout.output_matrix)
    write_dataset(g,"Wout_rev",Wout_rev.output_matrix)
    
    # Save reservoir matrix (use sparse format if applicable)
    gW = create_group(g,"reservoir_matrix")
    if issparse(esn.reservoir_matrix)
        I,J,X = findnz(sparse(esn.reservoir_matrix))  # Extract non-zero elements
        write_dataset(gW,"I",I)  # Row indices
        write_dataset(gW,"J",J)  # Column indices
        write_dataset(gW,"V",X)  # Values
    else
        write_dataset(gW,"W",esn.reservoir_matrix)
    end
    
    # Save input matrix (use sparse format if applicable)
    gWin = create_group(g,"input_matrix")
    if issparse(esn.input_matrix)
        I,J,X = findnz(sparse(esn.input_matrix))
        write_dataset(gWin,"I",I) 
        write_dataset(gWin,"J",J) 
        write_dataset(gWin,"V",X) 
    else        
        write_dataset(gWin,"Win",esn.input_layer)
    end

    # Save ESN type and ridge regression coefficient
    esn_type = string(typeof(esn))
    idx_name = 1:(findfirst('{',esn_type)-1)  # Extract type name without generic parameters
    attributes(g)["RC_type"]  = esn_type[idx_name]
    attributes(g)["Ridge_Coeff"] = exp(xopt[4])  # Store actual coefficient (not log)
    
    close(save_ESN)
end


# ==============================================================================
# MAIN CODE: Parameter Setup and Model Training
# ==============================================================================

length_valid = 200;
rnd_seed = 803;
sparsity = 0.006;
NW = 400;
N_el_run = 1000
wash = 500
Maxtime = 20000
N_datasets = 500

folder_name = "../Dataset/"
data_file = "duff_$(N_datasets)_training.h5"
esn_name = "ESN_train_eq_inf_$(N_datasets)_maxtime_$(Maxtime)_nn_1"
SR_lb = 0.01;α_lb = 0.0;σ_lb = 0.1; β_lb = 1e-12;
SR_ub = 2.0; α_ub = 1.0; σ_ub =5.0;  β_ub = 2.0;
lb = Array{Float64, 1}(undef,4); # lower bounds
ub = Array{Float64, 1}(undef,4); # upped bounds
lb = [SR_lb, α_lb, σ_lb,log(β_lb)];
ub = [SR_ub, α_ub, σ_ub,log(β_ub)];
x0 = (lb+ub)/2;
x0.= transform(x0);



# ------------------------------------------------------------------------------
# Step 1: Load Training and Validation Data
# ------------------------------------------------------------------------------
@info "reading the data"
train_data, labels_train, test_data,train_rev, labels_train_rev, N_run, n_valid = read_data(folder_name, data_file)
all_train_states = Array{Float64, 2}(undef, NW, N_run*(N_el_run-wash))

@info "starting computation for case: $(esn_name)"

# ------------------------------------------------------------------------------
# Step 2: Optimize ESN Hyperparameters using CMA-ES
# ------------------------------------------------------------------------------
@info "optimizing the ESN parameter"
result = minimize(obj, x0, 4;  # Optimize 4 parameters
        lower = zeros(5),       # Note: bounds mismatch (5 vs 4) - likely a bug
        upper = 10.0*ones(5),
        popsize = 15,           # Population size for CMA-ES
        verbosity = 1,          # Print optimization progress
        seed = rand(UInt),
        maxtime = Maxtime);

# Extract and transform optimal parameters
xopt = xbest(result);
xopt = untransform(xopt);

@info "optimization standard ESN success, the best parameters are:
    SR = $(@sprintf("%.3F",xopt[1])) ; leaky rate = $(@sprintf("%.3F",xopt[2]));
    σ = $(@sprintf("%.3F",xopt[3])); β = $(@sprintf("%.3E",exp(xopt[4])))"

# ------------------------------------------------------------------------------
# Step 3: Train ESN with Optimal Parameters (Forward Direction)
# ------------------------------------------------------------------------------
run_train_data = train_data[:, 1:N_el_run]
esn = ESN(run_train_data,2, NW;
    reservoir = rand_sparse(; radius = xopt[1], sparsity = sparsity),
    input_layer = weighted_init(;scaling = xopt[3]),
    reservoir_driver = RNN(leaky_coefficient = xopt[2]),
    nla_type = NLADefault(),washout = wash,rng = Random.seed!(rnd_seed));

# Collect reservoir states from first training run
all_train_states[:,1:(N_el_run-wash)] = esn.states

# Collect reservoir states from all remaining training runs
for i in 2:N_run
	filtered_block0_values = train_data[:, (i-1)*N_el_run+1:i*N_el_run]
	all_train_states[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = create_states(esn, filtered_block0_values, washout = wash)
end

# Train forward output weights using ridge regression
# Train forward output weights using ridge regression
Wout = train(StandardRidge(exp(xopt[4])), all_train_states, labels_train)

# ------------------------------------------------------------------------------
# Step 4: Train Backward Output Weights (Time-Reversed)
# ------------------------------------------------------------------------------
# Collect reservoir states for time-reversed training data
all_train_states_rev = Array{Float64, 2}(undef, NW, N_run*(N_el_run-wash))
for i in 1:N_run
	filtered_block0_values = train_rev[:, (i-1)*N_el_run+1:i*N_el_run]
	all_train_states_rev[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = create_states(esn, filtered_block0_values, washout = wash)
end

# Train backward output weights (for time-reversed dynamics)
# Train backward output weights (for time-reversed dynamics)
Wout_rev = train(StandardRidge(exp(xopt[4])), all_train_states_rev, labels_train_rev)

# ------------------------------------------------------------------------------
# Step 5: Save Trained Model to HDF5
# ------------------------------------------------------------------------------
@info "saving the ESN "
fun_save_ESN(esn_name)
@info "success saving the ESN "

