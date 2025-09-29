import numpy as np
import scipy.stats as stats


def steigers_z_test(r1, r2, n):
    # Calculate the Z-values for the correlations
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Calculate the test statistic (Z)
    z_stat = (z1 - z2) / np.sqrt((n - 3) * 2)

    # Compute the p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return z_stat, p_value


def fisher_r_to_z_test(r1, r2, n1, n2):
    # Convert the correlations to Fisher's Z
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Calculate the standard error
    se = np.sqrt((1 / (n1 - 3)) + (1 / (n2 - 3)))

    # Calculate the Z-test statistic
    z_stat = (z1 - z2) / se

    # Compute the p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return z_stat, p_value


# Example: Two correlations r1 and r2 from the same sample with n = 30 participants
r1 = 0.898
r2 = 0.936
n = 158

z_stat, p_value = steigers_z_test(r1, r2, n)
print(f"Steiger's Z-test: Z-statistic = {z_stat:.3f}, p-value = {p_value:.3f}")

z_stat, p_value = fisher_r_to_z_test(r1, r2, n, n)
print(f"Fisher's r-to-z Test: Z-statistic = {z_stat:.3f}, p-value = {p_value:.3f}")
