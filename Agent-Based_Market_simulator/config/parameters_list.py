# List of parameters for simulation

class parameters_list:
    
    #timeframes
    ticks_per_day = 15000
    num_of_days = 5

    # initial params
    #init_data = [99.82, 99.89, 100]
    init_data = [100.32, 100.328, 100.314, 100.269, 100.265, 100.260, 100.254, 100.240, 100.236, 100.233, 100.232, 100.136, 100.130, 100.120, 100.115, 100]
    #init_volume = [0, 0, 0]
    init_volume = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #best bid and ask prices
    bid_price = [100 for _ in range(len(init_volume))]
    ask_price = [100 for _ in range(len(init_volume))]
    # lists for volume_inbalance
    bid_volume = [1 for _ in range(len(init_volume))]
    ask_volume = [1 for _ in range(len(init_volume))]

    tick_size = 0.01
    #aggregate_value
    aggregate_value = 30
    close_price = []
    min_price = []
    max_price = []
    spread = []
    volume_data = []
    volume_inb = []
    macd = []
    macdsignal = []
    macdhist = []
    data_rsi = []
    data_obv = []

    
    # agent' group prob.
    delta_mm = 0.25#0.1
    delta_lc = 0.05#0.1
    delta_mr = 0.5#0.4
    delta_mt = 0.5#0.4
    delta_nt = 0.75#0.75
    delta_ddpg = 0.0#0.025
    
    # Market maker params.
    mm_v_min = 1
    mm_v_max = 200000
    mm_v = 1
    mm_w = 50#70
    
    # Liquidity consumer params
    lc_h_min = 1
    lc_h_max = 100000
    
    # Mean reversion params
    #mrt_v_mr = 1
    #mrt_v_mr = 1000
    mrt_v_mr = 1#00
    mrt_alpha = 0.94
    
    # Momentum trader params
    mt_nr = 5
    mt_k = 0.001
    #mt_k = 0.0001
    
    # Noise trader params
    b_or_s_prob = 0.5
    lmbda_m = 0.03
    lmbda_l = 0.54
    lmbda_c = 0.43
    
    #lmbda_m = 0.05
    #lmbda_l = 0.44
    #lmbda_c = 0.53
    
    mu_mo = 7
    omeg_mo = 0.1
    mu_lo = 8
    omeg_lo = 0.7
    #xmin_offfspr = 0.05
    xmin_offfspr = 0.005
    beta_offspr = 2.72
    
    save_folder = "params_search/params_grid_ddpg.csv"
    save_aggregate_folder = "params_search/params_grid_ddpg_aggr.csv"
