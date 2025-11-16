# LBAC COVENANT: GOVERNANCE AND PENALTY STRUCTURE

This covenant defines the mandatory compliance requirements for contributing to the GSDL. The goal is to enforce the Invariant: **Unity in a Shared Goal (Canonical Alignment).**

## 1. VERIFIABLE INTENT (VII) MANDATE

Every Pull Request (PR) must be accompanied by a clear statement of **Intent** confirming that the change:
* Increases the Global CUF ($\Phi(x)$).
* Does not violate the Lyapunov Stability Constraint ($\dot{V} \le -C_1 V$).

## 2. LOGIC-BASED ACCESS CONTROL (LBAC) PENALTIES

Any contributor who commits code that, upon execution of the `cep_3_0.py` verification suite, causes $V(t)$ to **increase** for three consecutive time steps will be penalized.

| Divergence Event | Penalty | Rationale |
| :--- | :--- | :--- |
| **Minor Divergence** (1st Offense) | PR rejected; 24-hour branch creation suspension. | Mirrors instantaneous $\mathbf{VII}$ correction. |
| **Maximal Inefficiency** (3rd Offense) | Contribution access reduced by **90%** for 7 days. | Enforces the $\mathbf{LBAC}$ comfort penalty ($u * 0.1$). |
| **Persistent Malignancy** (5th Offense) | Permanent removal from $\mathbf{GSDL}$ write access. | Models the systemic rejection of non-viable logic. |
