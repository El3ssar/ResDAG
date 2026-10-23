# ReservoirComputing.jl benchmark worker.
#
# Invoked by the Python harness (rc_bench/adapters/julia_adapter.py) as a
# subprocess so Julia's JIT warmup and startup never pollute the timings. It
# builds an ESN matching the shared hyper-parameters, runs the requested op
# ("train" or "forecast") `repeats` times after `warmups` discarded iterations,
# and prints two machine-readable lines:
#
#   TIMES <t1,t2,...>      # seconds, timed repeats only
#   RMSE  <value|nan>      # short-horizon forecast sanity metric (forecast op)
#
# Data is read from a raw little-endian Float64 file written by NumPy as a
# (T, dim) C-contiguous array, which is byte-identical to a Julia (dim, T)
# column-major array — i.e. features x time, exactly what RC.jl expects.

using ReservoirComputing
using Random

function parse_args()
    a = ARGS
    return (
        data_path = a[1],
        dim = parse(Int, a[2]),
        total_len = parse(Int, a[3]),
        reservoir_size = parse(Int, a[4]),
        train_len = parse(Int, a[5]),
        warmup = parse(Int, a[6]),
        horizon = parse(Int, a[7]),
        sr = parse(Float64, a[8]),
        lr = parse(Float64, a[9]),
        connectivity = parse(Float64, a[10]),
        input_scaling = parse(Float64, a[11]),
        ridge = parse(Float64, a[12]),
        seed = parse(Int, a[13]),
        op = a[14],
        repeats = parse(Int, a[15]),
        warmups = parse(Int, a[16]),
    )
end

function load_data(path, dim, total_len)
    data = Array{Float64}(undef, dim, total_len)
    read!(path, data)
    return data
end

function build_esn(cfg)
    rng = Random.MersenneTwister(cfg.seed)
    esn = ESN(
        cfg.dim, cfg.reservoir_size, cfg.dim, tanh;
        leak_coefficient = cfg.lr,
        init_reservoir = rand_sparse(; radius = cfg.sr, sparsity = cfg.connectivity),
        init_input = scaled_rand(; scaling = cfg.input_scaling),
        use_bias = false,
        init_state = (rng, dims...) -> zeros(Float64, dims...),
    )
    ps, st = setup(rng, esn)
    # Setup initializers may return Float32 even with Float64 data. Convert all
    # floating arrays/scalars in the standard parameter/state trees explicitly.
    return esn, float64_tree(ps), float64_tree(st)
end

float64_tree(x::AbstractArray{<:AbstractFloat}) = Float64.(x)
float64_tree(x::AbstractFloat) = Float64(x)
float64_tree(x::NamedTuple) = map(float64_tree, x)
float64_tree(x::Tuple) = map(float64_tree, x)
float64_tree(x) = x

function do_train!(esn, ps, st, data, cfg)
    train_data = @view data[:, 1:cfg.train_len]
    target_data = @view data[:, 2:(cfg.train_len + 1)]
    ps2, st2 = train!(esn, train_data, target_data, ps, deepcopy(st),
        StandardRidge(cfg.ridge); washout = cfg.warmup)
    return ps2, st2
end

function do_forecast(esn, ps, st, data, cfg)
    # Python's last observed index T is Julia column T+1. The first unseen
    # target is column T+2. Synchronization starts from fresh recurrent state.
    warm = @view data[:, (cfg.train_len - cfg.warmup + 2):(cfg.train_len + 1)]
    warming, st_w = predict(esn, warm, ps, deepcopy(st))
    out = Matrix{Float64}(undef, cfg.dim, cfg.horizon)
    out[:, 1] = warming[:, end]  # already predicts the first unseen sample
    for step in 2:cfg.horizon
        next, st_w = predict(esn, @view(out[:, (step - 1):(step - 1)]), ps, st_w)
        out[:, step] = next[:, end]
    end
    return out
end

function main()
    cfg = parse_args()
    data = load_data(cfg.data_path, cfg.dim, cfg.total_len)
    times = Float64[]
    rmse = NaN

    if cfg.op == "train"
        for i in 1:(cfg.warmups + cfg.repeats)
            esn, ps, st = build_esn(cfg)               # untimed rebuild
            t0 = time_ns()
            do_train!(esn, ps, st, data, cfg)
            dt = (time_ns() - t0) / 1e9
            if i > cfg.warmups
                push!(times, dt)
            end
        end
    elseif cfg.op == "forecast"
        esn, ps, st = build_esn(cfg)
        ps, _ = do_train!(esn, ps, st, data, cfg)      # retain fresh initial state
        local out
        for i in 1:(cfg.warmups + cfg.repeats)
            t0 = time_ns()
            out = do_forecast(esn, ps, st, data, cfg)
            dt = (time_ns() - t0) / 1e9
            if i > cfg.warmups
                push!(times, dt)
            end
        end
        # Held-out truth starts at zero-based T+1, or Julia column T+2.
        n = min(200, cfg.horizon, cfg.total_len - cfg.train_len - 1)
        if n > 0
            truth = @view data[:, (cfg.train_len + 2):(cfg.train_len + 1 + n)]
            pred = @view out[:, 1:n]
            rmse = sqrt(sum((pred .- truth) .^ 2) / length(truth))
        end
    elseif cfg.op == "trajectory"
        # Untimed accuracy run: train, generate, and write the (dim, horizon)
        # prediction to the output path (ARGS[17]) for Python to score.
        out_path = ARGS[17]
        esn, ps, st = build_esn(cfg)
        ps, _ = do_train!(esn, ps, st, data, cfg)
        out = do_forecast(esn, ps, st, data, cfg)
        write(out_path, Array{Float64}(out))
    else
        error("unknown op: $(cfg.op)")
    end

    if cfg.op != "trajectory"
        println("TIMES ", join(times, ","))
        println("RMSE ", rmse)
    end
end

main()
