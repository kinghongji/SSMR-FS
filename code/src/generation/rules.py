"""Generation and comparator rule metadata exported with the results."""
def generation_rules(config):
    common = {"temperature_limits": [-18, 18], "daily_values": "direct integer draws remain integers; continuous proposals use np.rint (ties-to-even), then clip to support",
              "raw_counts": {"scenario": 1800, "stratified": 1800, "risk": 1800},
              "deduplication": "distinct within each scenario / stratum; risk distinct across five risk categories; merged full sequences unique, retaining first occurrence in scenario,stratified,risk order",
              "master_seed": config.random_seed,
              "attempt_limit_per_constrained_path": config.source_pool.generation_attempt_limit,
              "interpretation": "proposal distributions plus explicit final constraints; conditionally accepted profiles need not remain iid",
              "reference_records_used": False,
              "category_interpretation": "source categories identify generation and acceptance rules; groups can overlap in realized temperature properties and counts are design quotas, not occurrence probabilities"}
    scenario = [
      {"category":"stable_frozen","n":"D[1,10]","rule":"c~D[-18,-1]; daily c+D[-1,1], rounded and clipped to [-18,-1]","predicate":"all days -18<=T<=-1"},
      {"category":"stable_chilled","n":"D[1,10]","rule":"c~D[0,4]; daily c+D[-1,1], rounded and clipped to [0,4]","predicate":"all days 0<=T<=4"},
      {"category":"slow_warming","n":"max(3,D[1,10])","rule":"start D[-18,10], end D[min(18,start+4),18]; linear baseline+N(0,0.7^2). Hold n, redraw parameters/noise until final predicate","predicate":"all adjacent differences>=0 and final>first"},
      {"category":"fast_warming","n":"max(2,D[1,10])","rule":"tau D[1,n-1]; low D[-18,8], high D[min(18,low+8),18]; switch centers; integer perturb D[-1,1]","predicate":"switch from a lower segment to a higher segment"},
      {"category":"recooling","n":"max(3,D[1,10])","rule":"tau D[1,n-2]; start D[-18,8], peak D[min(18,start+6),18], end D[-18,max(-18,peak-3)]; piecewise linear; no daily noise","predicate":"warming then cooling"},
      {"category":"high_frequency_fluctuation","n":"max(2,D[1,10])","rule":"low D[-18,8], high D[min(18,low+8),18]; daily alternation+integer perturb D[-2,2]","predicate":"daily low/high alternation"},
      {"category":"low_frequency_fluctuation","n":"max(3,D[1,10])","rule":"same low/high; block=max(2,floor(n/3)); alternate blocks+integer perturb D[-1,1]","predicate":"longer-segment alternation"},
      {"category":"late_warming","n":"max(2,D[1,10])","rule":"l D[1,min(3,n)]; tau=max(1,n-l); low D[-18,8], high D[min(18,low+6),18]; initial low; tail linspace(high-2,high,n-tau); one-point tail equals high-2","predicate":"initial stable segment, late increasing segment"},
      {"category":"cross_zero","n":"max(2,D[1,10])","rule":"tau D[1,n-1], negative center D[-18,-1], positive center D[1,18], D[-1,1] daily noise; reverse p=.5; hold n, redraw until actual crossing","predicate":"at least one adjacent negative/nonnegative sign transition"},
      {"category":"temperature_abuse","n":"D[1,10]","rule":"baseline D[-2,8]; k D[1,n] different random positions replaced with D[15,18]","predicate":"at least one day 15<=T<=18"},
    ]
    scenario_labels = {
      "stable_frozen":"Stable frozen (all days −18 to −1 °C)",
      "stable_chilled":"Stable chilled (all days 0 to 4 °C)",
      "slow_warming":"Gradual nondecreasing warming",
      "fast_warming":"Separated low-to-high step",
      "recooling":"Warming followed by recooling",
      "high_frequency_fluctuation":"Alternating daily high/low",
      "low_frequency_fluctuation":"Alternating longer high/low segments",
      "late_warming":"Late-stage warming",
      "cross_zero":"At least one 0 °C crossing",
      "temperature_abuse":"At least one 15–18 °C exposure day",
    }
    for row in scenario:
        row["count"] = config.source_pool.scenario_count_per_type
        row["display_name"] = scenario_labels[row["category"]]
    stratified = {"duration_layers": list(config.source_pool.time_layers),
       "temperature_bands": list(config.source_pool.temperature_layers),
       "trend_rules": list(config.source_pool.path_type_layers), "a_levels": list(config.source_pool.amplitude_layers),
       "strata": 180,"count_per_stratum":config.source_pool.stratified_count_per_stratum,
       "duration":"discrete uniform within layer intersected with n>=1 stable, n>=2 warming/cooling, n>=3 oscillating; hold n through rejection",
       "center_proposal":"continuous U(temp_low,temp_high)",
       "daily_perturbation":"independent proposal U(-a,a), before integer rounding; quantization can change deviation by up to 0.5 degree",
       "stable_baseline":"constant center",
       "warming_baseline":"s=max(-18,c-a-U(0,3)); e=min(18,c+a+U(1,5)); linspace(s,e,n)",
       "cooling_baseline":"s=min(18,c+a+U(1,5)); e=max(-18,c-a-U(0,3)); linspace(s,e,n)",
       "oscillating_baseline":"c+a*sin(i*pi+phase), i=0..n-1, phase~U(0,2*pi)",
       "a_baseline_role":"existing endpoint offsets and sinusoidal amplitude also retained; a now bounds the added daily perturbation in all rules",
       "acceptance":"final integers must contain >=1 value in target band AND satisfy nondecreasing/net warming, nonincreasing/net cooling, or >=1 sign change among nonzero daily differences. Stable means a constant baseline with added daily jitter, not all days in the target band. No arbitrary day overwrite"}
    risk = [
      {"category":"cold_boundary","n":"D[1,10]","rule":"daily integer -2..6 with normalized weights [1,1,2,3,4,5,5,4,2]/27","predicate":"all days -2..6"},
      {"category":"transition_4_8","n":"D[1,10]","rule":"daily N(6,3^2), round/clip; hold n and redraw all daily values until at least one 4<T<=8","predicate":"at least one actual 4<T<=8 day"},
      {"category":"above_8_risk","n":"D[1,10]","rule":"baseline N(11,4^2), round/clip; k=max(1,floor(n/3)) different random positions replaced by D[9,15]","predicate":"at least one actual 9..15 day; other days can fall outside this range"},
      {"category":"above_15_abuse","n":"D[2,10]","rule":"baseline D[-3,12]; k~D[ceil(.25*n),floor(.70*n)]; k distinct random positions replaced by D[15,18]","predicate":"25%..70% of actual days at 15..18; remaining days -3..12; exposed days need not be consecutive"},
      {"category":"freeze_thaw","n":"max(2,D[1,10])","rule":"alternating daily D[-18,-1] / D[0,12]; reverse p=.5","predicate":"daily negative/nonnegative transitions; no physical freeze-thaw inference"},
    ]
    risk_labels = {
      "cold_boundary":"−2 to 6 °C boundary range (all days)",
      "transition_4_8":"4<T≤8 °C exposure (at least one day)",
      "above_8_risk":">8 °C exposure (at least one 9–15 °C day)",
      "above_15_abuse":"15–18 °C exposure (25%–70% of days)",
      "freeze_thaw":"Daily 0 °C sign alternation",
    }
    for row in risk:
        row["count"] = config.source_pool.risk_pool_size//5
        row["display_name"] = risk_labels[row["category"]]
    baseline = {"SRS":"n~D[1,10], daily T~D[-18,18]; reject duplicate complete sequences",
       "Sobol":"scrambled 11D; generate 2^ceil(log2(N)) and use first N",
       "OA-LHS":"complete prime-q strength-2 OA-LHS q^2>=N, q>=d-1; random row permutation and first N; subset not claimed strength-2",
       "MaxPro":"scrambled random LHS; attempt 4N random single-coordinate row swaps, accept only strict reduction of pair penalty; floor differences 1e-14, cap exponent700, no early stop",
       "mapping":"Sobol/OA-LHS/MaxPro only: duration=floor(10*u)+1; T=rint(-18+36*u), clip; take first n daily coordinates",
       "dedup_and_supplement":"all unique complete sequences; supplement with unique SRS if needed"}
    return {"common":common,"scenario":scenario,"stratified":stratified,"risk":risk,"baselines":baseline}
