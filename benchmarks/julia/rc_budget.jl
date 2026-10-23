# Persistent worker for the explicit common-model, equal-time-budget track.
# Startup/package loading is paid once and reported separately by Python.
# Each FIT builds and trains once; FORECAST reuses its fitted readout with fresh
# recurrent memory. All candidate commands and file IPC are inside the budget.

include(joinpath(@__DIR__, "rc_bench.jl"))
using SparseArrays

function budget_model(dim, width, radius, leak, density, input_scale, bias_scale, seed;
        sparse = false)
    rng = MersenneTwister(seed)
    esn = ESN(dim, width, dim, tanh;
        leak_coefficient = leak,
        init_reservoir = (rng, dims...) -> rand_sparse(rng, Float64, dims...;
            radius = radius, sparsity = density, return_sparse = sparse),
        init_input = (rng, dims...) -> scaled_rand(rng, Float64, dims...;
            scaling = input_scale),
        use_bias = true,
        init_bias = (rng, n) -> vec(scaled_rand(rng, Float64, n, 1;
            scaling = bias_scale)),
        init_state = (rng, dims...) -> zeros(Float64, dims...),
    )
    ps, st = setup(rng, esn)
    return esn, float64_tree(ps), float64_tree(st)
end

function budget_server()
    esn = nothing
    ps = nothing
    fresh = nothing
    println("READY\t", VERSION, "\t", pkgversion(ReservoirComputing))
    flush(stdout)
    for line in eachline(stdin)
        fields = split(line, '\t')
        fields[1] == "CLOSE" && break
        started = time_ns()
        try
            if fields[1] == "FIT"
                path = fields[2]
                dim, count, width, washout = parse.(Int, fields[3:6])
                radius, leak, density, input_scale, bias_scale, alpha = parse.(Float64, fields[7:12])
                seed = parse(Int, fields[13])
                sparse = length(fields) >= 14 && fields[14] == "sparse"
                data = load_data(path, dim, count)
                build_started = time_ns()
                esn, ps, fresh = budget_model(dim, width, radius, leak, density,
                    input_scale, bias_scale, seed; sparse = sparse)
                build_seconds = (time_ns() - build_started) / 1e9
                fit_started = time_ns()
                cfg = (train_len = count - 1, warmup = washout, ridge = alpha)
                ps, _ = do_train!(esn, ps, fresh, data, cfg)
                fit_seconds = (time_ns() - fit_started) / 1e9
                println("OK\t", (time_ns() - started) / 1e9, "\t",
                    build_seconds, "\t", fit_seconds, "\t", Base.summarysize(ps),
                    "\t", Sys.maxrss())
            elseif fields[1] == "FORECAST"
                esn === nothing && error("FIT must precede FORECAST")
                path = fields[2]
                dim, count, horizon = parse.(Int, fields[3:5])
                data = load_data(path, dim, count)
                cfg = (dim = dim, train_len = 0, warmup = count,
                    horizon = horizon, forecast_origin = count)
                out = do_forecast(esn, ps, fresh, data, cfg)
                write(fields[6], out)
                println("OK\t", (time_ns() - started) / 1e9)
            else
                error("Unknown worker command")
            end
        catch exception
            message = replace(sprint(showerror, exception), '\n' => ' ', '\t' => ' ')
            println("ERROR\t", (time_ns() - started) / 1e9, "\t", message)
        end
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    budget_server()
end
