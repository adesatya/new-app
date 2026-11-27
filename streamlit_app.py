import streamlit as st
import pandas as pd
import pickle
import altair as alt
from pathlib import Path

try:
    import Orange
except ModuleNotFoundError:
    st.error(
        "Paket Orange tidak ditemukan. Pastikan `orange3==3.41.0` "
        "ada di requirements.txt, lalu jalankan ulang aplikasi."
    )
    st.stop()

st.title("Dashboard Analitik DJPb")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "02_realisasi_anggaran_klasifikasi.csv"
MODEL_PATH = BASE_DIR / "models" / "Best_model.pkcls"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


data = load_data()
model = None
if MODEL_PATH.exists():
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Model gagal dimuat: {exc}")
else:
    st.error(f"File model tidak ditemukan: {MODEL_PATH.name}")

tab_prediksi, tab_visualisasi = st.tabs(["Menu Prediksi", "Visualisasi"])

with tab_prediksi:
    st.header("Prediksi Realisasi 95%")
    st.markdown("Masukkan data input berikut dan klik tombol untuk melihat hasil prediksi.")

    col1, col2 = st.columns(2)
    with col1:
        jumlah_spm = st.number_input(
            "Jumlah SPM",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            format="%d",
        )
        revisi_dipa = st.number_input(
            "Revisi DIPA",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            format="%d",
        )
        deviasi_rpd_persen = st.number_input(
            "Deviasi RPD (%)",
            min_value=0.00,
            max_value=30.00,
            value=5.50,
            format="%.2f",
        )
    with col2:
        skor_ikpa = st.slider(
            "Skor IKPA",
            min_value=70.00,
            max_value=100.00,
            value=85.00,
            step=0.01,
            format="%.2f",
        )
        tipe_satker = st.selectbox(
            "Tipe Satker",
            ["Dekonsentrasi", "Kantor Daerah", "Kantor Pusat", "Tugas Pembantuan"],
        )

    if st.button("Prediksi", type="primary"):
        if model is None:
            st.error("Model belum tersedia.")
        else:
            tipe_values = [
                "Dekonsentrasi",
                "Kantor Daerah",
                "Kantor Pusat",
                "Tugas Pembantuan",
            ]
            tipe_indicators = [
                1.0 if tipe_satker == tipe_value else 0.0
                for tipe_value in tipe_values
            ]
            input_row = [
                float(jumlah_spm),
                float(revisi_dipa),
                float(deviasi_rpd_persen),
                float(skor_ikpa),
                *tipe_indicators,
            ]

            try:
                table = Orange.data.Table.from_list(model.domain, [input_row])
                prediction_index = int(model(table)[0])
                probabilities = model(table, model.Probs)[0]
                class_name = model.domain.class_var.values[prediction_index]
                confidence = float(probabilities[prediction_index])

                if class_name == "Ya":
                    st.success(f"Prediksi: {class_name} (keyakinan {confidence:.1%})")
                else:
                    st.warning(f"Prediksi: {class_name} (keyakinan {confidence:.1%})")
            except Exception as exc:
                st.error(f"Prediksi gagal dijalankan: {exc}")

with tab_visualisasi:
    st.header("Visualisasi Bubble Plot")
    st.markdown(
        "Grafik bubble menunjukkan skor IKPA terhadap deviasi RPD, dengan warna berdasarkan kolom `realisasi_tercapai_95persen`."
    )

    chart = (
        alt.Chart(data)
        .mark_circle(opacity=0.7, size=100)
        .encode(
            x=alt.X(
                "skor_ikpa",
                title="Skor IKPA",
                scale=alt.Scale(domain=[70, 100]),
            ),
            y=alt.Y("deviasi_rpd_persen", title="Deviasi RPD (%)"),
            color=alt.Color(
                "realisasi_tercapai_95persen:N",
                title="Realisasi Tercapai 95%",
                scale=alt.Scale(
                    domain=["Ya", "Tidak"],
                    range=["#2ca02c", "#d62728"],
                ),
            ),
            tooltip=[
                "kode_satker",
                "tipe_satker",
                "skor_ikpa",
                "deviasi_rpd_persen",
                "realisasi_tercapai_95persen",
            ],
        )
        .properties(width=900, height=550)
        .interactive()
    )
    st.altair_chart(chart, width=900)

    st.subheader("Histogram Skor IKPA")
    histogram = (
        alt.Chart(data)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X(
                "skor_ikpa",
                title="Skor IKPA",
                bin=alt.Bin(maxbins=20),
            ),
            y=alt.Y("count()", title="Jumlah Satker"),
            tooltip=[
                alt.Tooltip("skor_ikpa", title="Rentang Skor IKPA"),
                alt.Tooltip("count()", title="Jumlah Satker"),
            ],
        )
        .properties(width=900, height=350)
        .interactive()
    )
    st.altair_chart(histogram, width=900)