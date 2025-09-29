from scipy.stats import ttest_rel
#ENC
rmse1_1, rmse1_2, rmse1_3, rmse1_4, rmse1_5, rmse1_6, rmse1_7 = 2.43, 1.68, 1.62, 3.68, 3.82, 1.96, 2.24  # ENC
rmse1_1, rmse1_2, rmse1_3, rmse1_4, rmse1_5, rmse1_6, rmse1_7 = 3.47, 1.62, 3.02, 6.00, 6.80, 1.78, 2.34  # DEC
rmse2_1, rmse2_2, rmse2_3, rmse2_4, rmse2_5, rmse2_6, rmse2_7 = 3.47, 1.62, 3.02, 6.00, 6.80, 1.78, 2.34  # DEC
rmse2_1, rmse2_2, rmse2_3, rmse2_4, rmse2_5, rmse2_6, rmse2_7 = 3.38, 1.13, 1.58, 4.67, 3.87, 1.07, 2.04  # RB ENC

# whisper-small
small_rb_dec = [3.87, 1.12, 1.64, 5.74, 4.21, 1.12, 2.13]  # RB DEC
small_rb_enc = [3.63, 1.13, 1.68, 4.83, 3.89, 1.09, 2.06]  # RB ENC
small_dec = [3.48, 1.63, 3.77, 6.13, 6.95, 1.78, 2.37]  # DEC
small_enc = [2.51, 1.7,  1.78, 3.87, 3.87, 1.99, 2.29]  # ENC

# whisper-large-v3
large_rb_dec = [2.85, 0.75, 1.22, 4.83, 5.01, 0.77, 1.71]  # RB DEC
large_rb_enc = [3.19, 0.76, 1.2, 3.85, 4.46, 0.77, 1.65]  # RB ENC
large_dec = [4.35, 1.53, 2.67, 6.46, 6.79, 1.6,  2.27]  # DEC
large_enc = [2.7,  1.71, 1.98, 5.27, 4.09, 2.01, 2.49]  # ENC
# Your RMSEs (example values)
rmse_sys1 = [rmse1_1, rmse1_2, rmse1_3, rmse1_4, rmse1_5, rmse1_6, rmse1_7]
rmse_sys2 = [rmse2_1, rmse2_2, rmse2_3, rmse2_4, rmse2_5, rmse2_6, rmse2_7]

# Paired t-test
t_stat, p_value = ttest_rel(small_rb_dec + large_rb_dec, small_dec + large_dec)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")


# small:
# enc dec: t-statistic: -2.3507, p-value: 0.0570
# rb_enc rb_dec: t-statistic: -1.7250, p-value: 0.1353
# enc rb_enc: t-statistic: -0.1515, p-value: 0.8846
# enc rb_dec: t-statistic: -0.6807,  p-value: 0.5214
# dec rb_dec: t-statistic: 2.1276, p-value: 0.0775
# dec rb_enc: t-statistic: 2.6238, p-value: 0.0394

# large:
# enc dec: t-statistic: -1.7789, p-value: 0.1256
# rb_enc rb_dec: t-statistic: -1.0851, p-value: 0.3196
# enc rb_enc: t-statistic: 2.1879, p-value: 0.0713
# enc rb_dec:  t-statistic: 1.5788, p-value: 0.1655
# dec rb_dec: t-statistic: 6.6881, p-value: 0.0005
# dec rb_enc: t-statistic: 4.6983, p-value: 0.0033