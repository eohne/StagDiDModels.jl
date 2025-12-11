function quiet_reg(args...; kwargs...)
    with_logger(SimpleLogger(stderr, Logging.Warn)) do
        reg(args...; kwargs...)
    end
end