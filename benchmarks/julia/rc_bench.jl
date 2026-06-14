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
        use_bias = true,  # match reservoirpy / resdag, which use a reservoir bias
    )
    ps, st = setup(rng, esn)
    return esn, ps, st
end

function do_train!(esn, ps, st, data, cfg)
    train_data = @view data[:, 1:cfg.train_len]
    target_data = @view data[:, 2:(cfg.train_len + 1)]
    ps2, st2 = train!(esn, train_data, target_data, ps, st,
        StandardRidge(cfg.ridge); washout = cfg.warmup)
    return ps2, st2
end

function do_forecast(esn, ps, st, data, cfg)
    warm = @view data[:, 1:cfg.warmup]
    _, st_w = predict(esn, warm, ps, st)               # teacher-forced warmup
    out, _ = predict(esn, cfg.horizon, ps, st_w; initialdata = Vector(data[:, cfg.warmup]))
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
        ps, st = do_train!(esn, ps, st, data, cfg)     # train once, untimed
        local out
        for i in 1:(cfg.warmups + cfg.repeats)
            t0 = time_ns()
            out = do_forecast(esn, ps, st, data, cfg)
            dt = (time_ns() - t0) / 1e9
            if i > cfg.warmups
                push!(times, dt)
            end
        end
        # short-horizon RMSE vs ground truth (columns warmup+1 ..)
        n = min(200, cfg.horizon, cfg.total_len - cfg.warmup)
        if n > 0
            truth = @view data[:, (cfg.warmup + 1):(cfg.warmup + n)]
            pred = @view out[:, 1:n]
            rmse = sqrt(sum((pred .- truth) .^ 2) / length(truth))
        end
    elseif cfg.op == "trajectory"
        # Untimed accuracy run: train, generate, and write the (dim, horizon)
        # prediction to the output path (ARGS[17]) for Python to score.
        out_path = ARGS[17]
        esn, ps, st = build_esn(cfg)
        ps, st = do_train!(esn, ps, st, data, cfg)
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
