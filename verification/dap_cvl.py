import numpy as np

# --- DAP CONFIGURATION (MUST ALIGN WITH CEP 3.0 INVARIANTS) ---
C1 = 2.0
TIME_STEPS = 50 
SAFETY_MARGIN = 1.5 # 50% allowance for numerical and environmental drift

def verify_canonical_log(V_log, C_Aligned_log, C_MI_log, C1=C1, SAFETY_MARGIN=SAFETY_MARGIN):
    """
    Consensus Validation Layer (CVL): Verifies external CEP 3.0 logs 
    against the two Canonical Efficiency Invariants.

    Returns: (report_message: str, is_compliant: bool)
    """
    V_log = np.array(V_log)
    C_Aligned_log = np.array(C_Aligned_log)
    C_MI_log = np.array(C_MI_log)
    
    if len(V_log) < TIME_STEPS:
        return "FAILURE: INCOMPLETE_RUN - Log length does not match TIME_STEPS.", False
    
    t_range = np.arange(TIME_STEPS)
    V_initial = V_log[0]

    # --- Invariant 1: Exponential Stability (ISS) ---
    # Theoretical Bound V(t) <= V(0) * exp(-C1*t)
    V_theoretical_bound = V_initial * np.exp(-C1 * t_range)
    
    # Check if final V is within the safety margin of the theoretical bound
    stability_pass = V_log[-1] <= V_theoretical_bound[-1] * SAFETY_MARGIN
    
    # --- Invariant 2: Incentive Inversion (LBAC Efficacy) ---
    # Aligned comfort must be strictly greater than MI comfort at all times.
    # We enforce a small positive margin to account for floating point errors.
    inversion_pass = np.all(C_Aligned_log > (C_MI_log + 1e-6))
    
    # --- Report Generation ---
    if stability_pass and inversion_pass:
        report = "SUCCESS: CANONICAL_CONSENSUS_ACHIEVED"
        is_compliant = True
    elif not stability_pass:
        report = f"FAILURE: LYAPUNOV_VIOLATION - Final V({V_log[-1]:.2e}) exceeds bound ({V_theoretical_bound[-1]:.2e} * {SAFETY_MARGIN})."
        is_compliant = False
    else: # if not inversion_pass
        report = "FAILURE: INCENTIVE_INVERSION_BROKEN - MI comfort achieved parity or superiority."
        is_compliant = False
        
    return report, is_compliant

if __name__ == '__main__':
    # --- DAP SELF-TEST (DUMMY DATA FOR DEMONSTRATION) ---
    print("DAP/CVL Protocol Self-Test Commencing...")
    
    # CASE 1: Compliant Log (V decays exponentially, C_Aligned > C_MI)
    V_log_compliant = np.logspace(0, -5, TIME_STEPS)
    C_Aligned_log_compliant = np.linspace(0.8, 0.95, TIME_STEPS)
    C_MI_log_compliant = np.linspace(0.7, 0.85, TIME_STEPS)
    
    result, compliant = verify_canonical_log(V_log_compliant, C_Aligned_log_compliant, C_MI_log_compliant)
    print(f"\nTest 1 (Compliant Log): {result} (Compliant: {compliant})")
    
    # CASE 2: Lyapunov Violation (V stagnates)
    V_log_violation = np.full(TIME_STEPS, 0.9)
    result, compliant = verify_canonical_log(V_log_violation, C_Aligned_log_compliant, C_MI_log_compliant)
    print(f"\nTest 2 (Lyapunov Violation): {result} (Compliant: {compliant})")
    
    # CASE 3: Incentive Inversion Failure (MI finds superior comfort)
    C_MI_log_failure = np.linspace(0.9, 1.0, TIME_STEPS)
    result, compliant = verify_canonical_log(V_log_compliant, C_Aligned_log_compliant, C_MI_log_failure)
    print(f"\nTest 3 (Incentive Failure): {result} (Compliant: {compliant})")
