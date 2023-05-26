import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the machine learning model
model = pickle.load(open('randomforest_klasifikasi.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Mapping dictionary for QUALCODE
qualcode_mapping = {
    'CQ2': 0,
    'CQ2MP': 1,
    'CQ2MP1': 2,
    'CQ2PT': 3,
    'CQ2PTU': 4,
    'CQ2UN': 5,
    'CQ2UN1': 6,
    'CQ39PT': 7,
    'CQ3': 8,
    'CQ3EN': 9,
    'CQ4': 10,
    'CQ5': 11,
    'CQ8D': 12,
    'CQ39': 13,
    'CQ45': 14,
    'CQUN': 15,
    'CQUN1': 16,
    'CQUN3': 17,
    'CQUN4': 18,
    'CQUN8C': 19,
    'CQUN8H': 20,
    'CQUN91': 21,
    'CQUNX': 22,
    'CQX': 23,
    'CQ-Z': 24,
    'DDQ1': 25,
    'DQ': 26,
    'DQUN91': 27,
    'HR1': 28
    # Add more mappings as needed
}

# Mapping of SPEC values to numeric order
spec_mapping = {
    'COMMERCIAL QUALITY': 0,
    'JIS G 3101 SS400': 1,
    'JIS G 3113 SAPH370': 2,
    'JIS G 3113 SAPH440': 3,
    'JIS G 3131 SPHC': 4,
    'JIS G 3131 SPHD': 5,
    'JIS G 3132 SPHT1': 6,
    'JIS G 3132 SPHT2': 7,
    'JIS G3141': 8,
    'KSAPH270C': 9,
    'KSA 29H': 10,
    'KSA 37H': 11,
    'KSA 39H': 12,
    'KSA29': 13,
    'KSA37': 14,
    'KSAPH270C': 15,
    'KNSS-1D': 16,
    'MP 38': 17,
    'MP 390': 18,
    'MP 440': 19,
    'MP1A': 20,
    'MP38': 21,
    'MPIC': 22,
    'MPW 2': 23,
    'SECONDARY': 24,
    'SNI 07 3567': 25,
    'SP121BQ': 26,
    'SPC': 27,
    'SPCG': 28,
    'SPCK-6': 29,
    'SPHG 450': 30,
    'TS G3100G': 31,
    'TS G3101G SPH270COD': 32,
    'TS G3101G SPH270DOD': 33,
    'TS G3101G SPH440OD': 34,
    'YSH270C-OP': 35,
    'JSH270C': 36,
    'MS 1705:2003 SPHC': 371
}

def predict(QUALCODE, SPEC, THICK, WIDTH, WEIGHT):
    # Convert QUALCODE and SPEC to numeric values
    qualcode_numeric = qualcode_mapping.get(QUALCODE, -1)
    spec_numeric = spec_mapping.get(SPEC, -1)

    # Check if the input values are valid
    if qualcode_numeric == -1 or spec_numeric == -1:
        return "Hasil Prediksi Tidak Valid"

    # Scale the input features
    scaled_features = scaler.transform([[THICK, WIDTH, WEIGHT]])

    # Perform prediction using the loaded model
    prediction = model.predict([[qualcode_numeric, spec_numeric, scaled_features[0][0], scaled_features[0][1], scaled_features[0][2]]])

    # Map the prediction to the corresponding result
    if prediction[0] == 0:
        return "HEAVY"
    elif prediction[0] == 1:
        return "HRPO"
    elif prediction[0] == 2:
        return "LITE"
    elif prediction[0] == 3:
        return "MEDIUM"
    else:
        return "Hasil Prediksi Tidak Valid"

def main():
    st.title('Predictor App')
    
    # Input fields
    QUALCODE = st.selectbox('QUALCODE', list(qualcode_mapping.keys()))
    SPEC = st.selectbox('SPEC', list(spec_mapping.keys()))
    THICK = st.number_input('THICK')
    WIDTH = st.number_input('WIDTH')
    WEIGHT = st.number_input('WEIGHT')

    # Prediction button
    if st.button('Predict'):
        result = predict(QUALCODE, SPEC, THICK, WIDTH, WEIGHT)
        st.write('Prediction Result:', result)

if __name__ == '__main__':
    main()
