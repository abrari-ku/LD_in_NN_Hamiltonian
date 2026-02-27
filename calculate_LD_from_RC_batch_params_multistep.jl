using Pkg
# Pkg.activate("C:/Users/KU100060615/.julia/dev/ReservoirComputing")
Pkg.activate("/home/kunet.ae/100060615/RCfiles/NLS_project4/ReservoirComputing")
using HDF5,Adapt, SparseArrays
using LoggingExtras, Dates, Printf
using ReservoirComputing
using Base.Threads


# Parse command line arguments
function parse_arguments()
    # Default values
    defaults = Dict(
        "RCmodel" => "",
        "tend" => 4.0,
        "gamma" => 0.7,
        "step_mult" => 1
    )
    
    # Parse arguments
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--tend" && i < length(ARGS)
            defaults["tend"] = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--gamma" && i < length(ARGS)
            defaults["gamma"] = parse(Float64, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--step_mult" && i < length(ARGS)
            defaults["step_mult"] = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--help"
            println("Usage: julia calculate_LD_from_RC_batch_params.jl [RCmodel] [--tend VALUE] [--gamma VALUE] [--step_mult VALUE]")
            println("Arguments:")
            println("  RCmodel         : RC model name (first positional argument)")
            println("  --tend VALUE    : Integration time (default: 4.0)")
            println("  --gamma VALUE   : Distance parameter (default: 0.7)")
            println("  --step_mult VALUE: Step multiplier for coarser LD (default: 1)")
            println("  --help          : Show this help message")
            exit(0)
        else
            # First positional argument is RCmodel
            if defaults["RCmodel"] == ""
                defaults["RCmodel"] = ARGS[i]
            end
            i += 1
        end
    end
    
    # If no RCmodel provided, use filename
    if defaults["RCmodel"] == ""
        defaults["RCmodel"] = basename(@__FILE__)
    end
    
    return defaults
end

# Parse arguments
args = parse_arguments()
RCmodel = args["RCmodel"]
tend = args["tend"]
γ = args["gamma"]
step_mult = args["step_mult"]

@info "Starting with parameters: RCmodel=$RCmodel, tend=$tend, γ=$γ, step_mult=$step_mult"


# Extract components from filename
# parts = split(RCmodel, '_')
# size = parts[3]  # Gets 1000, 125, 250, or 500
# prefix = occursin("inside", RCmodel) ? "inside" : 
#          occursin("outside", RCmodel) ? "outside" : "base"

const date_format = "yyyy-mm-dd HH:MM:SS"

# Get SLURM job ID and array ID from environment variables
job_id = get(ENV, "SLURM_JOB_ID", "local")
array_id = get(ENV, "SLURM_ARRAY_TASK_ID", "0")

# Create log directory if it doesn't exist
mkpath("log")

# Create a log file with SLURM job and array IDs
log_file = "log/LD_calc_$(job_id)_$(array_id).log"
@info "Logging to file: $log_file"

logger = FormatLogger(log_file; append=true) do io, args
    println(io, args._module, " | ", "[", args.level,": ", Dates.format(now(), date_format), "] ", args.message)
end;
global_logger(logger)

num_threads = Threads.nthreads()
@info "Number of available threads: $num_threads"

# Function to log progress at regular intervals during threaded computation
function log_progress(counter::Threads.Atomic{Int64}, total::Int, desc::String, log_interval::Int=5000)
    completed = counter[]
    if completed % log_interval == 0 || completed == total
        progress_percent = round(completed / total * 100, digits=2)
        @info "$desc: $completed/$total ($progress_percent%)"
    end
end

# const date_format = "yyyy-mm-dd HH:MM:SS"
# logger = FormatLogger("out_LD.log"; append=true) do io, args
    # println(io, args._module, " | ", "[", args.level,": ", Dates.format(now(), date_format), "] ", args.message)
# end;
# global_logger(logger)

num_threads = Threads.nthreads()
@info "Number of available threads: $num_threads"

# function to read the RC model 
function read_model(model,test_data; leaky = 1.0)
    res_size = read_attribute(model,"res_size")
    if haskey(model["input_matrix"],"I")
        
        I1 = read(model["input_matrix"],"I")
        J1 = read(model["input_matrix"],"J")
        X1 = read(model["input_matrix"],"V")
        input_matrix = sparse(I1,J1,X1)
    else
        input_matrix = read(model["input_matrix"],keys(model["input_matrix"])[1])
    end
    
    if haskey(model["reservoir_matrix"],"I")
        I2 = read(model["reservoir_matrix"],"I")
        J2 = read(model["reservoir_matrix"],"J")
        X2 = read(model["reservoir_matrix"],"V")
        reservoir_matrix = sparse(I2,J2,X2,res_size,res_size)
    else
        reservoir_matrix= read(model["reservoir_matrix"],keys(model["reservoir_matrix"])[1])
    end
        
        
    W_out= read(model,"Wout")
	W_out_rev= read(model,"Wout_rev")

    esn = ESN(test_data, reservoir_matrix ,input_matrix;
            reservoir_driver = RNN(leaky_coefficient = α),washout = wash)
    return esn,W_out,W_out_rev
end

function distance_vector(u1,u2,p = 1)
    # d_a = (sum((abs.(u1[1:2]-u2[1:2])).^p))^(1/p)
    d_a = (sum((abs.(u1[1:2]-u2[1:2])).^p))
    # d_b = sum(abs.(angular_distance.(u1[2],u2[2])).^p)
    return d_a
end

nx = 400
ny = 400
xg = [LinRange(-1.5, 1.5,nx)...]
yg = [LinRange(-0.8, 0.8,ny)...]
# γ = 0.7
dt = 0.1
train_test_file = h5open("rnn_test_ld_w200.h5","r")
train_test_file_back = h5open("rnn_test_ld_w200_back.h5","r")
test1 = read(train_test_file,"test_1")
wash = 200;
RC_new_combined = h5open("ESN_data_test2.h5", "r");
QP_esn = RC_new_combined[RCmodel]
alp = read_attribute(QP_esn,"res_driver")
i1 = findlast(",",alp)
α = parse(Float64,alp[i1[1]+1:end-1])
esn_cycle_c,W_out_cycle_c,W_out_cycle_c_back = read_model(QP_esn,test1,leaky = α);

# tend = 4
Nt = Int(tend/dt) + 1
step = step_mult
# nstep = Int(tend/(dt*step)) + 1
group_name = "LD_eq_inf_$(RCmodel)_t_end_$(tend)_gamma_$(γ)_dtm_$(step_mult)_squared"
@info "group output name: $group_name"

@info "Model loaded, beginning computation at $(Dates.format(now(), date_format))"
start_time_fwd = time()
@info "Starting forward computation with $(160000) tests"
LD = Array{Float64}(undef,160000)
counter_fwd = Threads.Atomic{Int}(0)  # For tracking completion across threads

# @info "starting forward computation"
Threads.@threads for i in 1:160000
    test1 = read(train_test_file,"test_$i")
    predict_states = create_states(esn_cycle_c.reservoir_driver,test1 ,wash,esn_cycle_c.reservoir_matrix, esn_cycle_c.input_matrix,esn_cycle_c.bias_vector)
    UU= esn_cycle_c(Generative(Nt),W_out_cycle_c, predict_states[:,end])

    LD[i] = (step*dt)*sum([distance_vector(UU[:,k],UU[:,k+step],γ) for k in 1:step:Nt-step])
	# Update progress indicators
    Threads.atomic_add!(counter_fwd, 1)
    log_progress(counter_fwd, 160000, "Forward computation progress")
end
@info "end forward computation"
LD_reshaped = reshape(LD, 400, 400)



@info "Starting backward computation with $(160000) tests"
LDb = Array{Float64}(undef,160000)
counter_bwd = Threads.Atomic{Int}(0)  # For tracking completion across threads

Threads.@threads for i in 1:160000
    test1 = read(train_test_file_back,"test_$i")
    predict_states = create_states(esn_cycle_c.reservoir_driver,test1 ,wash,esn_cycle_c.reservoir_matrix, esn_cycle_c.input_matrix,esn_cycle_c.bias_vector)
    UU= esn_cycle_c(Generative(Nt),W_out_cycle_c_back , predict_states[:,end])
    UU= esn_cycle_c(Generative(Nt),W_out_cycle_c_back, predict_states[:,end])
    LDb[i] = (step*dt)*sum([distance_vector(UU[:,k],UU[:,k+step],γ) for k in 1:step:Nt-step])
	
	# Update progress indicators
    Threads.atomic_add!(counter_bwd, 1)
    log_progress(counter_bwd, 160000, "Backward computation progress")
end

@info "end backward computation, saving"
end_time = time()
total_runtime = end_time - start_time_fwd
@info "Total computation time: $(round(total_runtime, digits=2)) seconds ($(round(total_runtime/60, digits=2)) minutes)"

LDb_reshaped = reshape(LDb, 400, 400)

@info "Writing results to HDF5 file at $(Dates.format(now(), date_format))"
save_start = time()

# fid = h5open("LD_output_RC_t5.h5", "cw")
# group_name = "LD_"*RCmodel
fid = h5open("LD_output_RC_params.h5", "cw")
# group_name = "LD_$(RCmodel)_t_end_$(tend)_squared"
g = create_group(fid, group_name) 
# g = fid[group_name]
write_attribute(g,"Date", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
write_attribute(g,"xgrid", xg)
write_attribute(g,"ygrid", yg)
write_attribute(g,"t_end",tend)
write_attribute(g,"gamma",γ)
write_attribute(g,"step_mult",step_mult)
write_attribute(g,"dt",dt)
write_dataset(g,"LD_f",LD_reshaped)
write_dataset(g,"LD_b",LDb_reshaped)
close(fid)
save_end = time()
save_time = save_end - save_start
@info "Saving finished in $(round(save_time, digits=2)) seconds"