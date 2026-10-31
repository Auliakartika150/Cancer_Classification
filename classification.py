import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================
st.set_page_config(
    page_title="Breast Cancer | ML Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Warna utama dashboard
NAVY = "#17324D"
BLUE = "#2878B5"
TEAL = "#18A6A6"
PURPLE = "#8064A2"
GREEN = "#26966F"
ORANGE = "#E69A36"
RED = "#D95D5D"
LIGHT_BG = "#F4F7FB"
TEXT = "#243447"

# CSS untuk tampilan dashboard
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {LIGHT_BG};
    }}

    [data-testid="stHeader"] {{
        background: rgba(244, 247, 251, 0.9);
    }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1450px;
    }}

    /* Sidebar: latar biru gelap */
    [data-testid="stSidebar"] {{
        background: #17324D;
    }}

    /* Judul dan label sidebar tetap putih */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: #FFFFFF !important;
    }}

    /* Teks di dalam kotak input dan pilihan dibuat gelap */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] * {{
        color: #243447 !important;
    }}

    /* Teks bantuan pada sidebar */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: #DCE7F1 !important;
    }}

    .hero {{
        background: linear-gradient(115deg, #17324D 0%, #245C83 58%, #2878B5 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(23, 50, 77, 0.13);
    }}

    .hero h1 {{
        color: white;
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 8px 0;
    }}

    .hero p {{
        color: #E4EFF8;
        font-size: 14px;
        margin: 0;
    }}

    .section-title {{
        color: {NAVY};
        font-size: 20px;
        font-weight: 750;
        margin: 8px 0 4px 0;
    }}

    .section-subtitle {{
        color: #66788A;
        font-size: 13px;
        margin-bottom: 14px;
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #E5EBF2;
        padding: 17px 18px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(23, 50, 77, 0.045);
    }}

    div[data-testid="stMetricLabel"] p {{
        color: #65788B !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {NAVY} !important;
        font-weight: 800 !important;
    }}

    div[data-testid="stTabs"] button {{
        font-weight: 650;
        color: #52677B;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {BLUE} !important;
    }}

    .info-card {{
        background: white;
        border: 1px solid #E5EBF2;
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }}

    .info-card h4 {{
        color: {NAVY};
        margin: 0 0 8px 0;
        font-size: 15px;
    }}

    .info-card p {{
        color: #65788B;
        font-size: 13px;
        margin: 0;
        line-height: 1.6;
    }}

    .small-note {{
        color: #718096;
        font-size: 12px;
    }}

    /* Tombol dengan kontras teks yang jelas */
    .stButton > button,
    .stFormSubmitButton > button,
    [data-testid="stFormSubmitButton"] button {{
        background: {BLUE} !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 9px;
        font-weight: 700 !important;
        padding: 0.55rem 1rem;
    }}

    .stButton > button *,
    .stFormSubmitButton > button *,
    [data-testid="stFormSubmitButton"] button * {{
        color: #FFFFFF !important;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {{
        background: {NAVY} !important;
        color: #FFFFFF !important;
        border: 0 !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 2. MEMUAT DATASET LOKAL
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cancer_Data.csv"


@st.cache_data
def load_data():
    """Membaca dan membersihkan dataset kanker payudara."""
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    data = data.dropna(subset=["diagnosis"])
    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "File Cancer_Data.csv tidak ditemukan. "
        "Pastikan file CSV berada dalam folder yang sama dengan app_classification.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Dataset gagal dibaca: {error}")
    st.stop()

if "diagnosis" not in df.columns:
    st.error("Kolom 'diagnosis' tidak ditemukan dalam dataset.")
    st.stop()

# Pastikan fitur yang digunakan berupa angka
X = df.drop(columns=["diagnosis"]).apply(pd.to_numeric, errors="coerce")
y = df["diagnosis"].astype(str).str.strip().str.upper()

# Hapus baris yang memiliki nilai fitur kosong/tidak valid
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)
df = df.loc[valid_rows].reset_index(drop=True)
feature_names = X.columns.tolist()

if y.nunique() != 2:
    st.error("Aplikasi ini dirancang untuk klasifikasi dua kelas diagnosis.")
    st.stop()

# =========================================================
# 3. SIDEBAR: PENGATURAN MODEL
# =========================================================
with st.sidebar:
    st.markdown("## 🧬 ML Explorer")
    st.markdown("Pengaturan model dan evaluasi")
    st.divider()

    st.markdown("### Pengaturan data")
    test_size = st.slider(
        "Porsi data pengujian",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        help="Contoh 20% berarti 80% data untuk pelatihan dan 20% untuk pengujian.",
    )

    st.markdown("### Pengaturan SVM")
    c_value = st.select_slider(
        "Kekuatan regularisasi (C)",
        options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        value=1.0,
        help="Nilai C mengatur seberapa besar model memberi penalti pada kesalahan data training.",
    )
    kernel = st.selectbox(
        "Jenis kernel",
        options=["rbf", "linear"],
        format_func=lambda value: "RBF (nonlinear)" if value == "rbf" else "Linear",
    )
    seed = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    st.divider()
    st.markdown("**Informasi dataset**")
    st.caption(f"Nama file: {DATA_PATH.name}")
    st.caption(f"Jumlah data bersih: {len(df):,}")
    st.caption(f"Jumlah fitur: {len(feature_names)}")

# =========================================================
# 4. MEMBAGI DATA DAN MELATIH MODEL
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=int(seed),
    stratify=y,
)

# StandardScaler dipelajari hanya dari data training melalui Pipeline
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                C=float(c_value),
                kernel=kernel,
                probability=True,
                class_weight="balanced",
                random_state=int(seed),
            ),
        ),
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model_classes = list(model.named_steps["svm"].classes_)
positive_label = "M" if "M" in model_classes else model_classes[-1]
positive_index = model_classes.index(positive_label)
y_prob = model.predict_proba(X_test)[:, positive_index]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, pos_label=positive_label, zero_division=0
)
recall = recall_score(
    y_test, y_pred, pos_label=positive_label, zero_division=0
)
f1 = f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
auc = roc_auc_score((y_test == positive_label).astype(int), y_prob)

# =========================================================
# 5. HEADER DASHBOARD
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>Breast Cancer Classification</h1>
        <p>
            Dashboard analisis data dan evaluasi model Support Vector Machine (SVM).
            Pelajari pola data, lihat performa model, dan coba simulasi klasifikasi.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Ringkasan Performa Model</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Hasil evaluasi pada data pengujian yang tidak digunakan saat pelatihan.</div>',
    unsafe_allow_html=True,
)

# Kartu metrik
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{accuracy:.1%}", help="Persentase seluruh prediksi yang benar.")
m2.metric("Precision", f"{precision:.1%}", help="Ketepatan prediksi pada kelas malignant.")
m3.metric("Recall", f"{recall:.1%}", help="Kemampuan menemukan kasus malignant.")
m4.metric("F1-score", f"{f1:.3f}", help="Keseimbangan precision dan recall.")
m5.metric("ROC-AUC", f"{auc:.3f}", help="Kemampuan model membedakan dua kelas.")

st.markdown("")
tabs = st.tabs(
    [
        "Ikhtisar",
        "Evaluasi Model",
        "Eksplorasi Data",
        "Simulasi Prediksi",
    ]
)

# =========================================================
# 6. TAB IKHTISAR
# =========================================================
with tabs[0]:
    st.markdown(
        '<div class="section-title">Gambaran Dataset</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Ringkasan jumlah data dan komposisi kategori diagnosis.</div>',
        unsafe_allow_html=True,
    )

    benign_count = int((y == "B").sum())
    malignant_count = int((y == "M").sum())

    a, b, c, d = st.columns(4)
    a.metric("Total sampel", f"{len(df):,}")
    b.metric("Benign (jinak)", f"{benign_count:,}")
    c.metric("Malignant (ganas)", f"{malignant_count:,}")
    d.metric("Jumlah fitur", f"{len(feature_names)}")

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        dist = (
            y.value_counts()
            .rename_axis("Diagnosis")
            .reset_index(name="Jumlah")
        )
        dist["Kategori"] = dist["Diagnosis"].map(
            {"B": "Benign (Jinak)", "M": "Malignant (Ganas)"}
        )
        fig = px.pie(
            dist,
            names="Kategori",
            values="Jumlah",
            hole=0.62,
            color="Kategori",
            color_discrete_map={
                "Benign (Jinak)": TEAL,
                "Malignant (Ganas)": ORANGE,
            },
            title="Komposisi Diagnosis",
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hole=0.62,
        )
        fig.update_layout(
            height=390,
            margin=dict(l=15, r=15, t=60, b=15),
            legend_title_text="Kategori",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color=TEXT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        split_data = pd.DataFrame(
            {
                "Kelompok data": ["Training", "Testing"],
                "Jumlah": [len(X_train), len(X_test)],
            }
        )
        fig = px.bar(
            split_data,
            x="Kelompok data",
            y="Jumlah",
            color="Kelompok data",
            color_discrete_map={"Training": BLUE, "Testing": PURPLE},
            text="Jumlah",
            title="Pembagian Data Training dan Testing",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            height=390,
            showlegend=False,
            margin=dict(l=15, r=15, t=60, b=15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color=TEXT,
            yaxis_title="Jumlah data",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="info-card">
            <h4>Memahami hasil klasifikasi</h4>
            <p>
                <b>Benign</b> berarti jinak, sedangkan <b>Malignant</b> berarti ganas.
                Model mempelajari pola dari fitur numerik pada dataset untuk
                memperkirakan kategori diagnosis. Hasil ini merupakan demonstrasi
                machine learning, bukan keputusan medis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 7. TAB EVALUASI MODEL
# =========================================================
with tabs[1]:
    st.markdown(
        '<div class="section-title">Evaluasi Kinerja Model</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Periksa jenis prediksi yang benar dan kesalahan yang masih dibuat model.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Confusion Matrix")
        labels = ["B", "M"]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=["Aktual: Benign", "Aktual: Malignant"],
            columns=["Prediksi: Benign", "Prediksi: Malignant"],
        )
        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale=[
                [0, "#EAF2F8"],
                [0.5, "#78A9D1"],
                [1, NAVY],
            ],
            aspect="auto",
            labels=dict(x="Hasil prediksi", y="Kondisi sebenarnya", color="Jumlah"),
        )
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="white",
            font_color=TEXT,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Baris menunjukkan kondisi sebenarnya, sedangkan kolom menunjukkan hasil prediksi model."
        )

    with right:
        st.markdown("#### Kurva ROC")
        fpr, tpr, _ = roc_curve(
            (y_test == positive_label).astype(int),
            y_prob,
        )
        roc_df = pd.DataFrame(
            {
                "False Positive Rate": fpr,
                "True Positive Rate": tpr,
            }
        )
        fig = px.area(
            roc_df,
            x="False Positive Rate",
            y="True Positive Rate",
            title=f"ROC Curve | AUC {auc:.3f}",
        )
        fig.update_traces(
            line=dict(color=TEAL, width=3),
            fillcolor="rgba(24,166,166,0.13)",
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#9AA8B5", dash="dash"),
                name="Garis acuan",
            )
        )
        fig.update_layout(
            height=390,
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color=TEXT,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend_title_text="Keterangan",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Kurva yang mendekati sudut kiri atas umumnya menunjukkan kemampuan pemisahan kelas yang lebih baik."
        )

    st.markdown("#### Rincian hasil untuk setiap kelas")
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=["Benign (Jinak)", "Malignant (Ganas)"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.info(
        "Perhatikan nilai recall Malignant. Recall menunjukkan seberapa banyak "
        "sampel malignant yang berhasil ditemukan oleh model. False Negative "
        "perlu mendapat perhatian khusus dalam konteks klasifikasi medis."
    )

# =========================================================
# 8. TAB EKSPLORASI DATA
# =========================================================
with tabs[2]:
    st.markdown(
        '<div class="section-title">Kenali Karakteristik Data</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Pilih salah satu karakteristik untuk membandingkan pola pada dua kategori diagnosis.</div>',
        unsafe_allow_html=True,
    )

    feature = st.selectbox(
        "Pilih karakteristik sel tumor",
        options=feature_names,
        format_func=lambda value: value.replace("_", " ").title(),
    )

    plot_df = df.loc[X.index, [feature]].copy()
    plot_df["Diagnosis"] = y.values
    plot_df["Kategori"] = plot_df["Diagnosis"].map(
        {"B": "Benign (Jinak)", "M": "Malignant (Ganas)"}
    )

    fig = px.histogram(
        plot_df,
        x=feature,
        color="Kategori",
        barmode="overlay",
        opacity=0.68,
        marginal="box",
        color_discrete_map={
            "Benign (Jinak)": TEAL,
            "Malignant (Ganas)": ORANGE,
        },
        title=f"Sebaran nilai {feature.replace('_', ' ').title()}",
        labels={feature: feature.replace("_", " ").title()},
    )
    fig.update_layout(
        height=480,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color=TEXT,
        legend_title_text="Kategori diagnosis",
        margin=dict(l=15, r=15, t=60, b=15),
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = (
        plot_df.groupby("Kategori")[feature]
        .agg(["count", "mean", "median", "std"])
        .rename(
            columns={
                "count": "Jumlah data",
                "mean": "Rata-rata",
                "median": "Nilai tengah",
                "std": "Standar deviasi",
            }
        )
        .round(3)
    )
    st.markdown("#### Ringkasan nilai fitur")
    st.dataframe(summary, use_container_width=True)
    st.caption(
        "Grafik menunjukkan pola dalam dataset. Perbedaan sebaran tidak membuktikan hubungan sebab-akibat dan bukan batas diagnosis."
    )

# =========================================================
# 9. TAB SIMULASI PREDIKSI
# =========================================================
with tabs[3]:
    st.markdown(
        '<div class="section-title">Simulasi Klasifikasi</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Masukkan nilai karakteristik numerik untuk melihat hasil model.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "Gunakan fitur ini hanya untuk mencoba cara kerja model. "
        "Hasilnya bukan diagnosis dan tidak boleh digunakan untuk keputusan medis."
    )

    st.markdown(
        """
        <div class="info-card">
            <h4>Petunjuk pengisian</h4>
            <p>
                Setiap kolom mewakili satu karakteristik sel. Nilai awal diisi
                menggunakan nilai tengah dari dataset agar formulir dapat langsung
                dicoba. Ubah nilainya jika ingin menguji skenario lain.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.form("prediction_form"):
        values = {}
        input_cols = st.columns(3)

        for index, col_name in enumerate(feature_names):
            default_value = float(X[col_name].median())
            values[col_name] = input_cols[index % 3].number_input(
                col_name.replace("_", " ").title(),
                value=default_value,
                format="%.2f",
                help=f"Masukkan nilai untuk fitur {col_name}.",
            )

        submitted = st.form_submit_button(
            "Jalankan simulasi",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame([values], columns=feature_names)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        probability_by_class = {
            label: float(probability)
            for label, probability in zip(model_classes, probabilities)
        }

        st.markdown("### Hasil simulasi")
        if prediction == "M":
            st.error("Hasil model: Malignant (Ganas)")
        else:
            st.success("Hasil model: Benign (Jinak)")

        prob_cols = st.columns(len(model_classes))
        for index, label in enumerate(model_classes):
            label_text = "Benign (Jinak)" if label == "B" else "Malignant (Ganas)"
            prob_cols[index].metric(
                label_text,
                f"{probability_by_class[label]:.1%}",
            )

        probability_df = pd.DataFrame(
            {
                "Kategori": [
                    "Benign (Jinak)" if label == "B" else "Malignant (Ganas)"
                    for label in model_classes
                ],
                "Skor model": [
                    probability_by_class[label] for label in model_classes
                ],
            }
        )
        fig = px.bar(
            probability_df,
            x="Kategori",
            y="Skor model",
            color="Kategori",
            text=probability_df["Skor model"].map(lambda value: f"{value:.1%}"),
            color_discrete_map={
                "Benign (Jinak)": TEAL,
                "Malignant (Ganas)": ORANGE,
            },
            title="Perbandingan skor keluaran model",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis=dict(range=[0, 1.12], tickformat=".0%"),
            showlegend=False,
            height=350,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color=TEXT,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Skor yang ditampilkan berasal dari model SVM dan belum menjadi probabilitas klinis yang terkalibrasi."
        )

# =========================================================
# 10. INFORMASI METODOLOGI DAN FOOTER
# =========================================================
with st.expander("Metode analisis dan batasan aplikasi"):
    st.markdown(
        """
        **Tahapan analisis**
        1. Membaca dataset `Cancer_Data.csv` dari folder aplikasi.
        2. Menghapus kolom identitas dan kolom kosong yang tidak diperlukan.
        3. Memisahkan 30 fitur karakteristik sebagai data input dan `diagnosis` sebagai target.
        4. Membagi data menjadi training dan testing dengan pembagian stratified.
        5. Melakukan standardisasi menggunakan `StandardScaler`.
        6. Melatih model Support Vector Machine (SVM).
        7. Mengukur performa menggunakan Accuracy, Precision, Recall, F1-score, dan ROC-AUC.
        """
    )

st.markdown("")
st.markdown(
    """
    <div style="
        text-align:center;
        color:#7B8A99;
        font-size:12px;
        padding:18px 0 4px 0;
        border-top:1px solid #E2E8F0;">
        Breast Cancer ML Explorer · Educational Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)
