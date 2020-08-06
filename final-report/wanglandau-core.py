def simulation(system, Es,
                max_sweeps = 1_000_000,
                flat_sweeps = 10_000,
                eps = 1e-8,
                logf0 = 1,
                flatness = 0.2
               ):
    """
    Run a Wang-Landau simulation on system with energy bins Es to determine
    the system density of states g(E).
    
    Args:
        system: The system to perform the simulation on (see systems module).
        Es: The energy bins of the system to access. May be a subset of all bins.
        max_sweeps: The scale for the maximum number of MC sweeps per f-iteration.
            The actual maximum iterations may be fewer, but approaches max_sweeps
            exponentially as the algorithm executes. 
        flat_sweeps: The number of sweeps between checks for histogram flatness.
            In AJP [10.1119/1.1707017], Landau et. al. use 10_000 sweeps.
        eps: The desired tolerance in f. Wang and Landau [WL] use 1e-8 in the original
            paper [10.1103/PhysRevLett.86.2050].
        logf0: The initial value of ln(f). WL set to 1.
        flatness: The desired flatness of the histogram. WL set to 0.2 (80% flatness).
    
    Returns:
        A tuple of results with entries:
        Es: The energy bins the algorithm was passed.
        S: The logarithm of the density of states (microcanonical entropy).
        H: The histogram from the last f-iteration.
        converged: True if each f-iteration took fewer than the maximum sweeps.
    
    Raises:
        ValueError: One of the parameters was invalid.
    """
    if (max_sweeps <= 0
        or flat_sweeps <= 0
        or eps <= 1e-16
        or not (0 < logf0 <= 1)
        or not (0 <= flatness < 1)):
        raise ValueError('Invalid Wang-Landau parameter.')

    # Initial values
    M = max_sweeps * system.sweep_steps
    flat_iters = flat_sweeps * system.sweep_steps
    logf = 2 * logf0 # Compensate for first loop iteration
    logftol = np.log(1 + eps)
    converged = True
    steps = 0
    
    E0 = Es[0]
    Ef = Es[-1]
    N = len(Es) - 1
    S = np.zeros(N) # Set all initial g's to 1
    H = np.zeros(N, dtype=np.int32)
    i = binindex(Es, system.E)
    
    while logftol < logf:
        H[:] = 0
        logf /= 2
        iters = 0
        niters = int((M + 1) * np.exp(-logf / 2))
        while (iters % flat_iters != 0 or not flat(H, flatness)) and iters < niters:
            system.propose()
            Eν = system.Eν
            j = binindex(Es, Eν)
            if E0 <= Eν < Ef and (
                S[j] < S[i] or np.random.rand() <= np.exp(S[i] - S[j])):
                system.accept()
                i = j
            H[i] += 1
            S[i] += logf
            iters += 1
        steps += iters
        if niters <= iters:
            converged = False
    
    return Es, S, H, steps, converged

