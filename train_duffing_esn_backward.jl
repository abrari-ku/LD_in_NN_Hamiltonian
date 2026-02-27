using Pkg
Pkg.activate("/home/kunet.ae/100060615/RCfiles/NLS_project4/ReservoirComputing")
Pkg.resolve()
using Parameters,Random,CMAEvolutionStrategy,HDF5,Adapt, SparseArrays
using LoggingExtras, Dates, Printf
using ReservoirComputing

const date_format = "yyyy-mm-dd HH:MM:SS"
logger = FormatLogger("out2.log"; append=true) do io, args
    println(io, args._module, " | ", "[", args.level,": ", Dates.format(now(), date_format), "] ", args.message)
end;
global_logger(logger)
function read_data()
	folder_name = "/dpc/kuin0117/Abrari/Project4/GHNN/Data/"
    train_test_file = h5open(folder_name*data_file,"r")
    #reading  feature and labels for training
    train_test_data = read(train_test_file["features"],"block0_values")
    N_run = Int(size(train_test_data,2)/N_el_run)
    train_data = read(train_test_file["features"],"block0_values")
    labels = read(train_test_file["labels"],"block0_values") 
    labels_train = Array{Float64, 2}(undef,2, N_run*(N_el_run-wash))
    labels_train_rev = Array{Float64, 2}(undef,2, N_run*(N_el_run-wash))
    labels_rev = reverse(train_test_data,dims = 2)
    train_rev = reverse(labels,dims = 2)
    for i in 1:N_run
        labels_train[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)]= labels[:, (i-1)*N_el_run+wash+1:i*N_el_run]
        labels_train_rev[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = labels_rev[:, (i-1)*N_el_run+wash+1:i*N_el_run]
    end
    #reading data for testing
    test_set = read(train_test_file["test_features"],"block0_values")
    n_valid = Int(size(test_set,2)/N_el_run)
    test_data=Array{AbstractArray{Float64,2},1}(undef,n_valid)
    for i in 1:n_valid
        test_data[i] = test_set[:,(i-1)*N_el_run+1:(i-1)*N_el_run+wash+length_valid] 
    end
    close(train_test_file)
    return train_data, labels_train, test_data, train_rev, labels_train_rev, N_run, n_valid
end

function Loss(esn,Wout,last_state, validation_set)
    N = size(validation_set,2)
    output = esn(Generative(N),Wout, last_state = last_state)
    err = output - validation_set
    return sum(sqrt.(sum(abs2,err,dims = 1)))
end

function obj(θ)
    """Objective function to optimize"""

    esn, Wout = compute_esn_all_runs(θ)
    loss = zeros(n_valid);
    for i in 1:n_valid
        states_test = create_states(esn,test_data[i][:,1:wash+1], washout = wash);
        loss[i] = Loss(esn,Wout,states_test[:,1], test_data[i][:,wash+1:end]);
    end
    return sum(loss)/(n_valid*length_valid)
end

function compute_esn_all_runs(θ)
    θ = untransform(θ)
    SR, α, σ, logβ = θ;
    # empty!(all_train_states)
    run_train_data = train_data[:, 1:N_el_run]
    esn = ESN(run_train_data,2, NW;
        reservoir = rand_sparse(; radius = SR, sparsity = sparsity),
        input_layer = weighted_init(;scaling = σ),
        reservoir_driver = RNN(leaky_coefficient = α),
        nla_type = NLADefault(),washout = wash,rng = Random.seed!(rnd_seed));

        
    all_train_states[:,1:(N_el_run-wash)] = esn.states

    for i in 2:N_run
        filtered_block0_values = train_data[:, (i-1)*N_el_run+1:i*N_el_run]
        all_train_states[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = create_states(esn, filtered_block0_values, washout = wash)
    end

    Wout = train(StandardRidge(exp(logβ)), all_train_states, labels_train)

    return esn, Wout
end

function untransform(θ::AbstractArray{Float64, 1})
    return lb + (ub - lb).*θ/10
end

function transform(θ::Array{Float64, 1})
    return Adapt.adapt(typeof(θ), 10*(θ - lb)./(ub - lb))
end

function fun_save_ESN(esn_name)
    save_ESN = h5open("ESN_data_test2.h5","cw")
    g = create_group(save_ESN, esn_name)
    attributes(g)["res_size"] = esn.res_size
    attributes(g)["res_driver"] = string(esn.reservoir_driver)
    res_d_str = (string(esn.reservoir_driver))
    attributes(g)["nla_type"] = string(esn.nla_type)
    attributes(g)["states_type"] = string(esn.states_type)
    write_dataset(g,"Wout",Wout.output_matrix)
    # depth = size(esn.reservoir_matrix,1)
    # attributes(g)["depth"]  = depth
    write_dataset(g,"Wout_rev",Wout_rev.output_matrix)
    
    gW = create_group(g,"reservoir_matrix")
    if issparse(esn.reservoir_matrix)
        I,J,X = findnz(sparse(esn.reservoir_matrix))
        write_dataset(gW,"I",I) 
        write_dataset(gW,"J",J) 
        write_dataset(gW,"V",X)
    else
        write_dataset(gW,"W",esn.reservoir_matrix)
    end
    gWin = create_group(g,"input_matrix")
    if issparse(esn.input_matrix)
        I,J,X = findnz(sparse(esn.input_matrix))
        write_dataset(gWin,"I",I) 
        write_dataset(gWin,"J",J) 
        write_dataset(gWin,"V",X) 
    else        
        write_dataset(gWin,"Win",esn.input_layer)
    end

    esn_type = string(typeof(esn))
    idx_name = 1:(findfirst('{',esn_type)-1)
    attributes(g)["RC_type"]  = esn_type[idx_name]
    attributes(g)["Ridge_Coeff"] = exp(xopt[4])
    close(save_ESN)
end



############### Main Code ####################

# n_valid = 3;
length_valid = 200;
# train_length = 2000;
rnd_seed = 803;
sparsity = 0.006;
NW = 400;
N_el_run = 1000
# N_run = 8
wash = 500
# N_datasets = 500;
Maxtime = 20000

if length(ARGS) > 0
    N_datasets = parse(Int, ARGS[1])
else
    N_datasets = 500
end

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



@info "reading the data"
train_data, labels_train, test_data,train_rev, labels_train_rev, N_run, n_valid = read_data()
all_train_states = Array{Float64, 2}(undef, NW, N_run*(N_el_run-wash))

@info "starting computation for case: $(esn_name)"

@info "optimizing the ESN parameter"
result = minimize(obj, x0, 4;
        lower = zeros(5),
        upper = 10.0*ones(5),
        popsize = 15,
        verbosity = 1,
        seed = rand(UInt),
        maxtime = Maxtime);

xopt = xbest(result);
xopt = untransform(xopt);

@info "optimization standard ESN success, the best parameters are:
    SR = $(@sprintf("%.3F",xopt[1])) ; leaky rate = $(@sprintf("%.3F",xopt[2]));
    σ = $(@sprintf("%.3F",xopt[3])); β = $(@sprintf("%.3E",exp(xopt[4])))"
run_train_data = train_data[:, 1:N_el_run]
esn = ESN(run_train_data,2, NW;
    reservoir = rand_sparse(; radius = xopt[1], sparsity = sparsity),
    input_layer = weighted_init(;scaling = xopt[3]),
    reservoir_driver = RNN(leaky_coefficient = xopt[2]),
    nla_type = NLADefault(),washout = wash,rng = Random.seed!(rnd_seed));

all_train_states[:,1:(N_el_run-wash)] = esn.states

for i in 2:N_run
	filtered_block0_values = train_data[:, (i-1)*N_el_run+1:i*N_el_run]
	all_train_states[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = create_states(esn, filtered_block0_values, washout = wash)
end

Wout = train(StandardRidge(exp(xopt[4])), all_train_states, labels_train)
#Wout = train(esn,train_data[:,wash+1:end],StandardRidge(exp(xopt[4])))

all_train_states_rev = Array{Float64, 2}(undef, NW, N_run*(N_el_run-wash))
for i in 1:N_run
	filtered_block0_values = train_rev[:, (i-1)*N_el_run+1:i*N_el_run]
	all_train_states_rev[:,(i-1)*(N_el_run-wash)+1:i*(N_el_run-wash)] = create_states(esn, filtered_block0_values, washout = wash)
end
Wout_rev = train(StandardRidge(exp(xopt[4])), all_train_states_rev, labels_train_rev)


@info "saving the ESN "
fun_save_ESN(esn_name)
@info "success saving the ESN "

