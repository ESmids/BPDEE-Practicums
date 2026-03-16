import joblib
import pickle

# Laad de bestaande joblib bestanden
print("Laden van joblib bestanden...")
loaded_model = joblib.load("../regression_model.joblib")
loaded_scaler = joblib.load("../regression_scaler.joblib")

# Sla ze op als pickle bestanden
print("Opslaan als pickle bestanden...")
with open("regression_model.pkl", "wb") as f:
    pickle.dump(loaded_model, f)

with open("regression_scaler.pkl", "wb") as f:
    pickle.dump(loaded_scaler, f)

print("✓ regression_model.pkl aangemaakt")
print("✓ regression_scaler.pkl aangemaakt")
print("\nBestanden zijn klaar voor je Streamlit app!")

