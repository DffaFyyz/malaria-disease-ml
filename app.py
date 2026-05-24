from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

app = Flask(__name__)

# 1. LOAD MODEL BERSAMAAN SAAT SERVER MENYALA
print("Loading model...")
model = joblib.load("model/malaria_rf_model.pkl")
print("Model loaded successfully!")


# 2. FUNGSI EKSTRAKSI (Ini "Mata" servermu)
def extract_features_from_image(img_bgr):
    # --- 1. MASKING SEL ---
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh_cell = cv2.threshold(
        img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask_cell = cv2.bitwise_not(thresh_cell)

    # --- 2. MASKING PARASIT ---
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s_inside_cell = s[mask_cell == 255]

    if len(s_inside_cell) == 0:
        return [0, 0, 0, 0]  # Format array untuk model

    mean_s = np.mean(s_inside_cell)
    std_s = np.std(s_inside_cell)
    batas_parasit = mean_s + (3 * std_s) + 10

    _, mask_parasite_raw = cv2.threshold(s, batas_parasit, 255, cv2.THRESH_BINARY)
    mask_parasite_final = cv2.bitwise_and(mask_parasite_raw, mask_cell)

    # --- 3. CONTOURS ---
    contours_cell, _ = cv2.findContours(
        mask_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_parasite, _ = cv2.findContours(
        mask_parasite_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cell_area = 0
    largest_cell_cnt = None
    max_area = 0

    for cnt in contours_cell:
        area = cv2.contourArea(cnt)
        if area > 500:
            cell_area += area
            if area > max_area:
                max_area = area
                largest_cell_cnt = cnt

    parasite_area = sum(
        [cv2.contourArea(cnt) for cnt in contours_parasite if cv2.contourArea(cnt) > 25]
    )
    parasite_count = sum([1 for cnt in contours_parasite if cv2.contourArea(cnt) > 25])

    # --- 4. TEKSTUR (GLCM) ---
    texture_contrast = 0
    if largest_cell_cnt is not None:
        x, y, w, h = cv2.boundingRect(largest_cell_cnt)
        cell_crop = img_gray[y : y + h, x : x + w]
        if cell_crop.size > 0:
            glcm = graycomatrix(
                cell_crop,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True,
            )
            texture_contrast = graycoprops(glcm, "contrast")[0, 0]

    # Return urutan harus SAMA PERSIS dengan saat training!
    return [cell_area, parasite_count, parasite_area, texture_contrast]


# 3. ENDPOINT API (Menerima Request)
@app.route("/predict", methods=["POST"])
def predict():
    # Cek apakah ada file gambar yang dikirim
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected image"}), 400

    try:
        # Ajaibnya Flask: Membaca gambar langsung dari memori tanpa di-save ke hardisk
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Invalid image format"}), 400

        # 1. Gunakan "Mata" untuk mengekstrak 4 angka
        features = extract_features_from_image(img_bgr)

        # 2. Gunakan "Otak" (.pkl) untuk menebak dari 4 angka tersebut
        # Model scikit-learn butuh input array 2D, jadi kita bungkus dengan list [features]
        nama_kolom = [
            "cell_area",
            "parasite_count",
            "parasite_area",
            "texture_contrast",
        ]
        features_df = pd.DataFrame([features], columns=nama_kolom)

        # Prediksi menggunakan DataFrame
        prediction = model.predict(features_df)
        probabilities = model.predict_proba(features_df)

        # probabilities mengembalikan array 2D, contoh: [[0.12, 0.88]] (Artinya: 12% Sehat, 88% Sakit)
        # Kita ambil angka probabilitas dari tebakan yang menang
        predicted_class = prediction[0]
        confidence_score = probabilities[0][predicted_class]

        # Mapping hasil angka menjadi teks (0 = Sehat, 1 = Sakit)
        result_text = "Parasitized" if predicted_class == 1 else "Uninfected"

        # Kembalikan response JSON ke Backend Express/Frontend
        return jsonify(
            {
                "status": "success",
                "prediction": result_text,
                "confidence": round(float(confidence_score), 4),
                "extracted_features": {
                    "cell_area": features[0],
                    "parasite_count": features[1],
                    "parasite_area": features[2],
                    "texture_contrast": features[3],
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Jalankan server di port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
