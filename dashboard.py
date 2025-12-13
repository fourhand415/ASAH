import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import gc 

# --- Page Config ---
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide", page_icon="🛍️")

# --- Global Configuration ---
CLUSTER_PROFILE = {
    0: {
        "name": "At-Risk Low Spenders", 
        "icon": "🛒", 
        "desc": "Kelompok pelanggan yang rutin berbelanja namun dengan nilai keranjang (basket size) kecil. Sangat sensitif terhadap harga dan promosi kompetitor.", 
        "risk_profile": "Low Risk",
        "objective": "Increase Basket Size (Naikkan Nilai Transaksi)",
        "marketing_angle": "Value Maximization & Bundling Strategy"
    },
    1: {
        "name": "Champions / VIP", 
        "icon": "👑", 
        "desc": "Aset terbesar perusahaan. Pelanggan dengan frekuensi belanja tinggi, nominal transaksi besar, dan interaksi yang baru saja terjadi.", 
        "risk_profile": "Very Low Risk",
        "objective": "Retention & Delight (Jaga Loyalitas)",
        "marketing_angle": "Exclusivity, Pride & Service Excellence"
    },
    2: {
        "name": "Potential Loyalists", 
        "icon": "👥", 
        "desc": "Pelanggan dengan performa rata-rata yang menunjukkan sinyal positif. Memiliki potensi besar menjadi VIP jika diberikan insentif yang relevan.", 
        "risk_profile": "Medium Risk",
        "objective": "Upselling & Nurturing (Acelerate to VIP)",
        "marketing_angle": "Membership Laddering & Recommendation"
    },
    3: {
        "name": "Lost / Dead Customers", 
        "icon": "💤", 
        "desc": "Mantan pelanggan aktif yang sudah lama tidak kembali. Berisiko tinggi churn permanen jika tidak segera diintervensi.", 
        "risk_profile": "High Risk",
        "objective": "Reactivation / Win-Back (Penyelamatan Aset)",
        "marketing_angle": "Aggressive Discount, Urgency & FOMO"
    }
}

# --- Load Data ---
@st.cache_data
def load_basic_data():
    try:
        df_full = pd.read_csv("df_full.csv")
        rfm = pd.read_pickle("rfm.pkl")
        return df_full, rfm
    except: return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_recommendation_models():
    try:
        user_item_matrix = pd.read_pickle("user_item_matrix.pkl")
        item_similarity_df = pd.read_pickle("item_similarity_df.pkl")
        with open("topN_cluster.pkl", "rb") as f:
            topN_cluster = pickle.load(f)
        try:
            user_item_matrix.index = pd.to_numeric(user_item_matrix.index, errors="coerce").astype('Int64')
        except: pass
        return topN_cluster, user_item_matrix, item_similarity_df
    except: return None, None, None

df_full, rfm = load_basic_data()

# --- Helper Functions ---
def try_cast_customerid_to_int(df, col):
    try: df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64')
    except: pass
    return df

try:
    if not rfm.empty: rfm = try_cast_customerid_to_int(rfm, "Customer ID")
    if not df_full.empty: df_full = try_cast_customerid_to_int(df_full, "Customer ID")
except: pass

def normalize_customer_id_input(cid_input):
    try: return int(cid_input)
    except: return cid_input

def recommend_products(customer_id, rfm_df, top_cluster_df, user_matrix, item_sim_df, n=5):
    cid = normalize_customer_id_input(customer_id)
    try: in_rfm = cid in rfm_df['Customer ID'].astype(object).values
    except: in_rfm = cid in rfm_df['Customer ID'].values

    if not in_rfm: return None, f"ID {customer_id} tidak ditemukan."

    try: in_user_matrix = cid in user_matrix.index.astype(object)
    except: in_user_matrix = cid in user_matrix.index

    if not in_user_matrix: return None, f"ID {customer_id} tidak memiliki transaksi."

    try: cluster = int(rfm_df.loc[rfm_df['Customer ID'] == cid, 'Cluster'].values[0])
    except: cluster = int(rfm_df.loc[rfm_df['Customer ID'].astype(str) == str(cid), 'Cluster'].values[0])

    cluster_reco = top_cluster_df[top_cluster_df['Cluster'] == cluster]['Description'].tolist()[:n]
    
    bought_items = []
    try:
        bought_series = user_matrix.loc[cid]
        bought_items = bought_series[bought_series > 0].index.tolist()
    except:
        try:
            bought_series = user_matrix.loc[str(cid)]
            bought_items = bought_series[bought_series > 0].index.tolist()
        except: bought_items = []

    similar_items = []
    if len(bought_items) > 0:
        last_item = bought_items[-1]
        if last_item in item_sim_df.columns:
            sim = item_sim_df[last_item].sort_values(ascending=False).head(n+1).index.tolist()
            similar_items = [i for i in sim if i != last_item][:n]
    
    not_bought = [i for i in cluster_reco if i not in bought_items][:n]

    return {
        "Cluster": int(cluster),
        "Top Cluster Products": cluster_reco,
        "Similar Products (CF)": similar_items,
        "Cluster Products Not Bought": not_bought,
        "Bought List": bought_items
    }, None

# --- Sidebar ---
st.sidebar.image("https://assets.cdn.dicoding.com/original/commons/logo-asah.png", use_container_width=True)
l1, l2 = st.sidebar.columns(2)
with l1: st.image("https://assets.cdn.dicoding.com/original/commons/certificate_logo.png", use_container_width=True)
with l2: st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/250px-Accenture.svg.png", use_container_width=True)

st.sidebar.header("Global Filters")
if not df_full.empty:
    all_countries = sorted(df_full["Country"].dropna().unique())
    all_months = sorted(df_full["YearMonth"].dropna().unique())
    all_clusters = sorted(df_full["Cluster"].dropna().unique())

    country_filter = st.sidebar.multiselect("🌍 Filter Country", options=all_countries)
    month_filter = st.sidebar.multiselect("📅 Filter Bulan", options=all_months)
    cluster_filter = st.sidebar.multiselect("📦 Filter Cluster", options=all_clusters)

    df_filtered = df_full.copy()
    if country_filter: df_filtered = df_filtered[df_filtered["Country"].isin(country_filter)]
    if month_filter: df_filtered = df_filtered[df_filtered["YearMonth"].isin(month_filter)]
    if cluster_filter: df_filtered = df_filtered[df_filtered["Cluster"].isin(cluster_filter)]
else:
    df_filtered = pd.DataFrame()

menu = st.sidebar.radio("Navigasi:", ["Dashboard EDA", "Customer Recommendation", "Cluster Insight"])

# --- 1. Dashboard EDA ---
if menu == "Dashboard EDA":
    st.title("📈 Executive Dashboard Overview")
    st.markdown("Ringkasan performa bisnis makro berdasarkan parameter filter yang dipilih.")
    
    if df_filtered.empty:
        st.warning("Data kosong.")
    else:
        total_rev = df_filtered['Revenue'].sum()
        total_trx = df_filtered.shape[0]
        active_cust = df_filtered['Customer ID'].nunique()
        unique_prod = df_filtered['Description'].nunique()
        avg_sales = total_rev / total_trx if total_trx > 0 else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.metric("Total Revenue", f"£{total_rev:,.0f}", help="Total pendapatan kotor.")
        with k2: st.metric("Active Customers", f"{active_cust:,}", help="Jumlah customer unik.")
        with k3: st.metric("Total Transactions", f"{total_trx:,}", help="Total volume transaksi.")
        with k4: st.metric("Unique Products", f"{unique_prod:,}", help="Jumlah varian produk.")
        with k5: st.metric("Avg. Sales / Trx", f"£{avg_sales:,.2f}", help="Rata-rata nilai keranjang belanja.")

        with st.expander("📂 Klik untuk melihat Sampel Data Transaksi"):
            st.dataframe(df_filtered.head(), use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📅 Revenue Trend Analysis")
            rev_trend = df_filtered.groupby("YearMonth")["Revenue"].sum()
            if not rev_trend.empty: st.line_chart(rev_trend, color="#29b5e8")
            else: st.warning("Data tren tidak tersedia.")

        with c2:
            st.subheader("👥 Cluster Distribution")
            c_dist = df_filtered.groupby("Customer ID")["Cluster"].first().value_counts().sort_index()
            c_dist.index = [f"{i}: {CLUSTER_PROFILE[i]['name'].split('(')[0]}" for i in c_dist.index]
            if not c_dist.empty: st.bar_chart(c_dist, color="#ffaa00")

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("🏆 Top 10 Best Sellers (Volume)")
            top_prod = df_filtered.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
            st.bar_chart(top_prod, horizontal=True)

        with c4:
            st.subheader("📍 Customer Value Map (RFM Segments)")
            active_c = df_filtered["Customer ID"].unique()
            rfm_f = rfm[rfm["Customer ID"].isin(active_c)].copy()
            if not rfm_f.empty:
                rfm_f["Cluster Group"] = rfm_f["Cluster"].map(lambda x: CLUSTER_PROFILE.get(x, {}).get('name', str(x)))
                st.scatter_chart(rfm_f, x="Recency", y="Monetary", color="Cluster Group", size="Frequency", height=350)

        st.markdown("---")
        st.subheader("💎 Top 10 High Value Customers (Champions)")
        top_c = df_filtered.groupby("Customer ID")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
        top_c = top_c.merge(rfm[['Customer ID', 'Cluster']], on='Customer ID', how='left')
        top_c['Cluster Group'] = top_c['Cluster'].map(lambda x: CLUSTER_PROFILE.get(x, {}).get('name', str(x)))
        top_c["Revenue"] = top_c["Revenue"].apply(lambda x: f"£{x:,.0f}")
        top_c['Customer ID'] = top_c['Customer ID'].astype(str)
        st.dataframe(top_c[['Customer ID', 'Cluster Group', 'Revenue']], use_container_width=True, hide_index=True)

        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data (CSV)", csv, "filtered_retail_data.csv", "text/csv")

# --- 2. Customer Recommendation ---
elif menu == "Customer Recommendation":
    st.header("🎯 Customer 360° & Recommendation Engine")
    st.caption("Modul analisis personal untuk tim Sales/Marketing dalam menentukan pendekatan taktis per individu.")

    with st.expander("💡 Cheat Sheet: Contoh ID Customer per Cluster (Untuk Demo)"):
        cols_cheat = st.columns(4)
        for i, cls_id in enumerate(sorted(rfm['Cluster'].unique())):
            with cols_cheat[i]:
                st.markdown(f"**{CLUSTER_PROFILE[cls_id]['name'].split('(')[0]}**")
                top3 = rfm[rfm['Cluster'] == cls_id].sort_values('Monetary', ascending=False).head(3)
                for _, r in top3.iterrows():
                    st.code(f"{int(r['Customer ID'])}")
                    st.caption(f"Rev: £{r['Monetary']:,.0f}")

    col_input, col_btn = st.columns([3, 1])
    with col_input: cid_input = st.number_input("Masukkan Customer ID Target", min_value=1, step=1)
    with col_btn:
        st.write("##") 
        check_btn = st.button("🔍 Generate Strategy", type="primary")

    def display_product_cards(p_list, df_src, label=None):
        if not p_list:
            p_list = df_src.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(5).index.tolist()
            label = "🔥 Global Best Seller"
        
        subset = df_src[df_src['Description'].isin(p_list)].copy()
        if subset.empty: return

        if 'UnitPrice' not in subset.columns:
            subset = subset[subset['Quantity'] > 0]
            subset['UnitPrice'] = subset['Revenue'] / subset['Quantity']

        stats = subset.groupby('Description').agg({'UnitPrice': 'mean', 'Quantity': 'sum'}).reset_index()
        stats = stats.set_index('Description').reindex(p_list).reset_index()
        cols = st.columns(2)
        
        for idx, row in stats.iterrows():
            if pd.isna(row['UnitPrice']): continue
            with cols[idx % 2]:
                with st.container(border=True):
                    if label: st.markdown(f":red-background[**{label}**]")
                    c1, c2 = st.columns([1, 4])
                    with c1: st.write("# 🛒")
                    with c2:
                        st.markdown(f"**{row['Description']}**")
                        st.markdown(f"Harga Rata-rata: **£{row['UnitPrice']:,.2f}**")
                        st.caption(f"Total Terjual: {int(row['Quantity']):,} unit")

    if check_btn:
        with st.spinner("Menganalisis profil, menghitung skor RFM, & mencari produk relevan..."):
            topN, u_matrix, i_sim = load_recommendation_models()
        
        res, err = recommend_products(cid_input, rfm, topN, u_matrix, i_sim)

        if err: st.error(err)
        else:
            cid = normalize_customer_id_input(cid_input)
            cluster = res["Cluster"]
            c_prof = CLUSTER_PROFILE.get(cluster, {})
            
            rec_df = rfm.loc[rfm["Customer ID"] == cid]
            my_rec, my_freq, my_mon = rec_df["Recency"].values[0], rec_df["Frequency"].values[0], rec_df["Monetary"].values[0]

            if my_rec <= 30: status, color = "Active 🟢", "green"
            elif my_rec <= 90: status, color = "Warning 🟡", "orange"
            else: status, color = "Churn Risk 🔴", "red"

            st.markdown("---")
            # Profile Header
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.title(c_prof['icon'])
                    st.markdown(f"### {c_prof['name']}")
                    st.caption(f"ID: {cid} | Status Kesehatan: :{color}[{status}]")
                with c2:
                    max_r, max_f, max_m = rfm['Recency'].max(), rfm['Frequency'].quantile(0.95), rfm['Monetary'].quantile(0.95)
                    def prog(v, m): return min((v/m), 1.0)
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Recency (Hari Terakhir)", f"{int(my_rec)} Hari")
                        st.progress(prog(max_r - my_rec, max_r))
                    with m2:
                        st.metric("Frequency (Transaksi)", f"{int(my_freq)}x")
                        st.progress(prog(my_freq, max_f))
                    with m3:
                        st.metric("Monetary (Total Belanja)", f"£{my_mon:,.0f}")
                        st.progress(prog(my_mon, max_m))

            # Strategy Brief
            st.markdown("### 📝 Executive Strategy Brief")
            with st.container(border=True):
                st.info(f"**Primary Marketing Objective:** {c_prof['objective']}")
                s1, s2 = st.columns(2)
                promo_txt = ""
                
                with s1:
                    st.markdown("**🛠️ Action Plan (Langkah Taktis):**")
                    if cluster == 1:
                        st.write("1. **Retention:** Berikan akses 'Early Bird' atau Pre-order untuk koleksi terbaru.")
                        st.write("2. **Appreciation:** Kirim kartu ucapan personal atau gift premium (bukan diskon).")
                        st.write("3. **Service:** Pastikan jalur komplain/layanan prioritas (Fast Track).")
                        promo_txt = "💎 VIP ACCESS ONLY"
                    elif cluster == 3:
                        st.write("1. **Reactivation:** Kirim email/WA otomatis dengan subjek 'We Miss You'.")
                        st.write("2. **Incentive:** Berikan voucher diskon agresif (20-30%) dengan batas waktu (Urgency).")
                        st.write("3. **Relevance:** Tampilkan produk Best Seller yang terbukti laku keras.")
                        promo_txt = "🏷️ COMEBACK SPECIAL 25%"
                    elif cluster == 0:
                        st.write("1. **Basket Building:** Tawarkan paket Bundling (Beli 3 Lebih Hemat).")
                        st.write("2. **Threshold:** Berikan Gratis Ongkir dengan minimum pembelian tertentu.")
                        st.write("3. **Push:** Gunakan notifikasi Flash Sale untuk produk harga miring.")
                        promo_txt = "⚡ BUNDLE SAVER DEAL"
                    else:
                        st.write("1. **Cross-sell:** Rekomendasikan produk pelengkap dari pembelian terakhir.")
                        st.write("2. **Lock-in:** Ajak bergabung ke Loyalty Program untuk kumpulkan poin.")
                        st.write("3. **Nurture:** Berikan insentif poin ganda untuk transaksi berikutnya.")
                        promo_txt = "✨ MEMBER EXCLUSIVE"
                
                with s2:
                    st.markdown("**💬 Communication Angle & Scripting:**")
                    if cluster == 1: 
                        st.caption("Tone: Apresiatif, Elegan, Eksklusif, Tidak 'Jualan'.")
                        st.write("🗣️ *'Halo Kak [Nama], terima kasih telah menjadi pelanggan setia kami. Sebagai apresiasi, kami ingin Kakak menjadi orang pertama yang memiliki koleksi ini...'*")
                    elif cluster == 3: 
                        st.caption("Tone: Emosional, To-the-point, Mendesak (FOMO).")
                        st.write("🗣️ *'Halo Kak! Sudah lama tidak mampir, kami rindu. Ada voucher spesial khusus buat Kakak yang akan hangus dalam 24 jam...'*")
                    elif cluster == 0: 
                        st.caption("Tone: Hemat, Value-oriented, Rasional.")
                        st.write("🗣️ *'Sayang banget ongkirnya Kak! Tambah 1 barang lagi biar dapat Gratis Ongkir dan lebih hemat lho...'*")
                    else: 
                        st.caption("Tone: Helpful, Advisory, Encouraging.")
                        st.write("🗣️ *'Hai Kak, barang yang Kakak beli kemarin akan sangat cocok jika dipadukan dengan item ini. Cek rekomendasinya yuk...'*")

            # Product Catalog
            st.write("### 📦 Curated Product Recommendations")
            t1, t2, t3 = st.tabs(["🔥 Top Segment Picks", "🤝 Personal Match (AI)", "🆕 Upsell Opportunities"])
            with t1: 
                st.caption(f"Produk paling populer yang dibeli oleh segmen **{c_prof['name']}**.")
                display_product_cards(res["Top Cluster Products"], df_full, promo_txt)
            with t2: 
                st.caption("Rekomendasi personal berdasarkan pola kemiripan belanja (Collaborative Filtering).")
                display_product_cards(res["Similar Products (CF)"], df_full, "❤️ FOR YOU")
            with t3: 
                st.caption("Produk populer di segmen ini yang **belum pernah** dibeli customer (Peluang Cross-sell).")
                display_product_cards(res["Cluster Products Not Bought"], df_full, "🆕 TRY THIS")
            
            # --- DESCRIPTIVE IMPACT ANALYSIS ---
            st.markdown("---")
            st.subheader("🚀 Simulation & Business Impact Analysis")
            st.caption("Analisis proyeksi dampak bisnis jika strategi rekomendasi di atas berhasil dieksekusi.")
            
            with st.container(border=True):
                col_imp_l, col_imp_r = st.columns([1, 2])
                
                with col_imp_l:
                    st.markdown("#### 🎯 Skenario Sukses")
                    scenario_text = ""
                    if cluster == 0: scenario_text = "Customer memanfaatkan penawaran **Bundling / Minimum Order**."
                    elif cluster == 1: scenario_text = "Customer melakukan pembelian produk **Pre-order / Exclusive Launch**."
                    elif cluster == 3: scenario_text = "Customer menukarkan voucher **Win-Back** sebelum kedaluwarsa."
                    else: scenario_text = "Customer menambahkan produk **Rekomendasi (Cross-sell)** ke keranjang."
                    
                    st.info(f"**Trigger Action:**\n{scenario_text}")

                with col_imp_r:
                    st.markdown("#### 📈 Analisis Dampak pada Metrik RFM")
                    if cluster == 0: # Low Spender
                        st.success("✅ **Monetary Surge:** Nilai rata-rata transaksi akan meningkat signifikan di atas baseline historis mereka.")
                        st.write("✅ **Efficiency:** Biaya logistik per unit turun karena pengiriman terkonsolidasi (bundling).")
                    elif cluster == 1: # Champions
                        st.success("✅ **Recency Lock:** Menjaga skor Recency tetap 'Hijau' (Fresh), mencegah customer melirik kompetitor.")
                        st.write("✅ **Lifetime Value (LTV):** Memperpanjang usia loyalitas customer (Retention) tanpa mengorbankan margin profit (tanpa diskon).")
                    elif cluster == 3: # Lost
                        st.success("✅ **Churn Reversal:** Dampak paling kritis adalah mereset status dari 'Churn Risk' menjadi 'Active'.")
                        st.write("✅ **Recency Recovery:** Skor Recency akan membaik drastis (dari ratusan hari menjadi 0), membuka peluang engagement baru.")
                    else: # Potential
                        st.success("✅ **Frequency Boost:** Meningkatkan frekuensi kunjungan yang akan membentuk kebiasaan (habit).")
                        st.write("✅ **Segment Upgrade:** Mengakumulasi nilai belanja untuk mempercepat transisi customer menuju tier 'Champion/VIP'.")

# --- 3. Cluster Insight ---
elif menu == "Cluster Insight":
    st.header("🔎 Cluster Intelligence & Persona Deep Dive")
    st.caption("Memahami DNA perilaku, demografi, dan preferensi produk setiap segmen pelanggan.")

    # Selectbox di atas
    c_opts = sorted(df_full["Cluster"].unique())
    sel_c = st.selectbox("🎯 Pilih Segmen untuk Dianalisis:", c_opts, format_func=lambda x: f"{x} - {CLUSTER_PROFILE[x]['name']}")
    
    # Filter Data
    c_df = df_full[df_full["Cluster"] == sel_c]
    c_rfm = rfm[rfm["Cluster"] == sel_c]
    c_prof = CLUSTER_PROFILE.get(sel_c, {})

    # Data Tambahan untuk Persona
    top_country = c_df['Country'].mode()[0] if not c_df.empty else "-"
    avg_basket = c_df.groupby("Invoice")['Revenue'].sum().mean()
    
    # --- PERSONA CARD (EXPANDED) ---
    with st.container(border=True):
        # Header & Icon
        c_head1, c_head2 = st.columns([0.5, 4])
        with c_head1: st.title(c_prof['icon'])
        with c_head2: 
            st.markdown(f"### {c_prof['name']}")
            st.caption(f"Risk Profile: **{c_prof['risk_profile']}**")

        st.markdown("---")
        
        # Kolom Deskriptif (Storytelling)
        col_story, col_psycho = st.columns([1.5, 1])
        
        with col_story:
            st.markdown("#### 📝 Deskripsi Segmen")
            st.write(c_prof['desc'])
            
            st.markdown("#### 🛒 Pola Belanja Khas")
            if sel_c == 0: # Low Spender
                st.write("Customer ini cenderung membeli barang satuan dengan harga murah. Mereka sering menunda checkout (abandoned cart) jika ongkos kirim dirasa mahal. Jarang membeli produk *full-price*.")
            elif sel_c == 1: # Champions
                st.write("Customer ini tidak ragu memborong banyak item dalam satu invoice. Mereka membeli varian produk terbaru dan sering melakukan *repeat order* tanpa menunggu momen diskon besar.")
            elif sel_c == 2: # Potential
                st.write("Customer ini sedang dalam fase 'eksplorasi'. Mereka sudah percaya dengan brand (terbukti dari transaksi > 1x), namun nilai belanja mereka belum maksimal. Butuh dorongan sedikit lagi.")
            elif sel_c == 3: # Lost
                st.write("Dahulu customer ini aktif, namun aktivitasnya terhenti total. Kemungkinan besar mereka sudah berpindah ke kompetitor atau tidak lagi membutuhkan kategori produk ini.")

        with col_psycho:
            st.markdown("#### 🧠 Psikografis (Mindset)")
            if sel_c == 0:
                st.info("**Motivasi:** Penghematan (Saving).\n\n**Hambatan:** Biaya Layanan/Ongkir.\n\n**Kata Kunci:** 'Murah', 'Diskon', 'Hemat'.")
            elif sel_c == 1:
                st.success("**Motivasi:** Kualitas & Status.\n\n**Hambatan:** Pelayanan lambat/Stok habis.\n\n**Kata Kunci:** 'Exclusive', 'New', 'Priority'.")
            elif sel_c == 2:
                st.warning("**Motivasi:** Validasi & Trust.\n\n**Hambatan:** Bingung memilih varian.\n\n**Kata Kunci:** 'Rekomendasi', 'Best Seller'.")
            elif sel_c == 3:
                st.error("**Motivasi:** Nostalgia/Penawaran Gila.\n\n**Hambatan:** Lupa brand/Kecewa.\n\n**Kata Kunci:** 'Miss You', 'Comeback'.")

        st.markdown("---")
        
        # 4 Key Metrics Bar
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("👥 Total Populasi", f"{c_rfm.shape[0]:,} User", help="Jumlah user dalam segmen ini")
        with k2: st.metric("💰 Kontribusi Revenue", f"£{c_df['Revenue'].sum():,.0f}", help="Total uang yang masuk dari segmen ini")
        with k3: st.metric("💳 Avg. Basket Size", f"£{avg_basket:,.2f}", help="Rata-rata nilai belanja per sekali transaksi (Invoice)")
        with k4: st.metric("🌍 Domisili Dominan", top_country, help="Negara asal mayoritas user")

    # --- Bagian Bawah (Sama seperti sebelumnya) ---
    g_rec, g_freq, g_mon = rfm["Recency"].mean(), rfm["Frequency"].mean(), rfm["Monetary"].mean()
    a_rec, a_freq, a_mon = c_rfm["Recency"].mean(), c_rfm["Frequency"].mean(), c_rfm["Monetary"].mean()

    st.subheader("📊 Behavioral DNA (vs Global Average)")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric("Rata-rata Recency", f"{a_rec:.1f} Hari", f"{a_rec - g_rec:.1f} vs Global", delta_color="inverse")
        st.caption(f"Global Avg: {g_rec:.1f} Hari")
        st.progress(min(a_rec / (g_rec * 2), 1.0))

    with m2:
        st.metric("Rata-rata Frequency", f"{a_freq:.1f} Trx", f"{a_freq - g_freq:.1f} vs Global")
        st.caption(f"Global Avg: {g_freq:.1f} Trx")
        st.progress(min(a_freq / (g_freq * 2), 1.0))

    with m3:
        d_mon = a_mon - g_mon
        d_str = f"-£{abs(d_mon):,.0f}" if d_mon < 0 else f"+£{abs(d_mon):,.0f}"
        st.metric("Rata-rata Total Belanja (LTV)", f"£{a_mon:,.0f}", f"{d_str} vs Global")
        st.caption(f"Global Avg: £{g_mon:,.0f}")
        st.progress(min(a_mon / (g_mon * 2), 1.0))

    st.markdown("---")
    st.subheader("📈 Revenue Performance Trend")
    trend = c_df.groupby("YearMonth")["Revenue"].sum()
    if not trend.empty: st.area_chart(trend, color="#3b8ed0", height=300)

    st.write("##")
    st.subheader("🏆 Product Preference (Top 5 Most Purchased)")
    top_i = c_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(5).reset_index()
    for i, r in top_i.iterrows():
        c_p, c_b = st.columns([2, 3])
        with c_p: st.write(f"**{i+1}. {r['Description']}**")
        with c_b: st.progress(r['Quantity'] / top_i['Quantity'].max(), text=f"{int(r['Quantity']):,} unit")

    st.markdown("---")
    st.subheader("🚀 Strategic Marketing Playbook")
    with st.container(border=True):
        st.info(f"**Objective:** {c_prof['objective']}")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**📢 Channel Focus:**")
            if sel_c == 1: st.write("Personal WA/Email (Human Touch), Exclusive Events.")
            elif sel_c == 3: st.write("Retargeting Ads, Email Automation, Push Notif.")
            elif sel_c == 0: st.write("Social Media, In-App Banner, Flash Sale Page.")
            else: st.write("Newsletter Mingguan, Loyalty App, Receipt Message.")
        with s2:
            st.markdown("**💬 Key Message:**")
            if sel_c == 1: st.write("'Exclusive', 'Priority', 'Early Access', 'Thank You'.")
            elif sel_c == 3: st.write("'We Miss You', 'Limited Time', 'Special Gift'.")
            elif sel_c == 0: st.write("'Hemat', 'Diskon', 'Gratis Ongkir', 'Paket'.")
            else: st.write("'Rekomendasi', 'Upgrade', 'Poin', 'Benefit'.")
        with s3:
            st.markdown("**⛔ What to Avoid:**")
            if sel_c == 1: st.write("Spamming, menawarkan barang murahan, respon lambat.")
            elif sel_c == 3: st.write("Membiarkan > 30 hari tanpa kontak (Churn Permanen).")
            elif sel_c == 0: st.write("Menghapus diskon tiba-tiba, min. order terlalu tinggi.")
            else: st.write("Mengabaikan potensi mereka (Silent Growth).")
