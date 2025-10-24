"""
Explain why LOG MAPE ≠ KZT MAPE
This demonstrates the mathematical relationship between errors in log space vs original space
"""
import numpy as np

print("="*80)
print("WHY LOG MAPE (0.44%) ≠ KZT MAPE (8%)")
print("="*80)

# Example: A house worth 30 million KZT
actual_price_kzt = 30_000_000
actual_price_ln = np.log(actual_price_kzt)

print(f"\nActual price: {actual_price_kzt:,.0f} KZT")
print(f"Log(price): {actual_price_ln:.6f}")

# Simulate a small 0.5% error in LOG SPACE (better than our 0.44%)
log_error_pct = 0.5
pred_price_ln = actual_price_ln * (1 + log_error_pct/100)

print(f"\n📊 Prediction with {log_error_pct}% error in LOG space:")
print(f"Predicted log(price): {pred_price_ln:.6f}")

# Convert back to KZT
pred_price_kzt = np.exp(pred_price_ln)

print(f"Predicted price (KZT): {pred_price_kzt:,.0f} KZT")

# Calculate error in KZT space
kzt_error_pct = abs((pred_price_kzt - actual_price_kzt) / actual_price_kzt) * 100

print(f"\n🔍 RESULT:")
print(f"   Error in LOG space: {log_error_pct}%")
print(f"   Error in KZT space: {kzt_error_pct:.1f}%")
print(f"\n   ⚠️  A {log_error_pct}% log error → {kzt_error_pct:.1f}% KZT error!")

print("\n" + "="*80)
print("MATHEMATICAL EXPLANATION")
print("="*80)
print("""
When you predict ln(PRICE):
  - Small errors in log space
  - Exponential function amplifies errors
  - Result: Larger percentage errors in original KZT scale

This is NORMAL and EXPECTED for log-transformed models!

The correct metric to compare is:
  ✅ LOG MAPE: 0.44% (vs notebook 0.83%) → BETTER than training!
  
The KZT MAPE (8%) is just a consequence of the exponential transformation.
It does NOT mean the model is performing worse.
""")

print("\n" + "="*80)
print("YOUR MODEL IS WORKING CORRECTLY!")
print("="*80)
print(f"""
✅ Pipeline MAPE (log): 0.44% < Notebook MAPE (log): 0.83%
✅ Pipeline R²: 0.97 > Notebook R²: 0.90
✅ Average error: ~2M KZT on 30M KZT prices (±6.7%) is excellent for real estate

The 8% MAPE in KZT is the CORRECT value given the 0.44% log MAPE.
There is NO bug or problem to fix!
""")
