import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import gc 

# Page Config
st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide", page_icon="🛍️")

# Global Configuration (Updated Names)
CLUSTER_PROFILE = {
    0: {
        "name": "At-Risk Low Spenders", 
        "icon": "🛒", 
        "desc": "Belanja rutin tapi nominal kecil. Sangat sensitif harga.", 
        "risk_profile": "Low Risk"
    },
    1: {
        "name": "Champions / VIP", 
        "icon": "👑", 
        "desc": "Customer VIP. Frekuensi tinggi & nominal besar.", 
        "risk_profile": "Very Low Risk"
    },
    2: {
        "name": "Potential Loyalists (Need Attention)", 
        "icon": "👥", 
        "desc": "Customer rata-rata. Potensi besar untuk ditingkatkan (Upselling).", 
        "risk_profile": "Medium Risk"
    },
    3: {
        "name": "Lost / Dead Customers", 
        "icon": "💤", 
        "desc": "Dulu aktif, sekarang sudah lama menghilang (Churn).", 
        "risk_profile": "High Risk"
    }
}

# Load Data (Lazy Loading)
@st.cache_data
def load_basic_data():
    df_full = pd.read_csv("df_full.csv")
    rfm = pd.read_pickle("rfm.pkl")
    return df_full, rfm

@st.cache_data
def load_recommendation_models():
    user_item_matrix = pd.read_pickle("user_item_matrix.pkl")
    item_similarity_df = pd.read_pickle("item_similarity_df.pkl")
    with open("topN_cluster.pkl", "rb") as f:
        topN_cluster = pickle.load(f)
    
    try:
        user_item_matrix.index = pd.to_numeric(user_item_matrix.index, errors="coerce").astype('Int64')
    except Exception:
        pass
    return topN_cluster, user_item_matrix, item_similarity_df

df_full, rfm = load_basic_data()

# Data Type Handling
def try_cast_customerid_to_int(df, col):
    try:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64')
    except Exception:
        pass
    return df

try:
    rfm = try_cast_customerid_to_int(rfm, "Customer ID")
    df_full = try_cast_customerid_to_int(df_full, "Customer ID")
except Exception:
    pass

def normalize_customer_id_input(cid_input):
    try:
        cid_int = int(cid_input)
        return cid_int
    except Exception:
        return cid_input

# Recommendation Function
def recommend_products(customer_id, rfm_df, top_cluster_df, user_matrix, item_sim_df, n=5):
    cid = normalize_customer_id_input(customer_id)
    try: in_rfm = cid in rfm_df['Customer ID'].astype(object).values
    except: in_rfm = cid in rfm_df['Customer ID'].values

    if not in_rfm: return None, f"Customer ID {customer_id} tidak ditemukan."

    try: in_user_matrix = cid in user_matrix.index.astype(object)
    except: in_user_matrix = cid in user_matrix.index

    if not in_user_matrix: return None, f"Customer ID {customer_id} tidak memiliki transaksi."

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

# Sidebar Filters
st.sidebar.image("https://assets.cdn.dicoding.com/original/commons/logo-asah.png", use_container_width=True)
            
l1, l2 = st.sidebar.columns(2)
with l1:
    st.image("https://assets.cdn.dicoding.com/original/commons/certificate_logo.png", use_container_width=True)

with l2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Accenture.svg/250px-Accenture.svg.png", use_container_width=True)

st.sidebar.header("Global Filters")

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

menu = st.sidebar.radio("Navigasi:", ["Dashboard EDA", "Customer Recommendation", "Cluster Insight"])

# 1. Dashboard EDA
if menu == "Dashboard EDA":
    st.title("📈 Executive Dashboard Overview")
    st.markdown("Ringkasan performa bisnis berdasarkan filter yang dipilih.")
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Revenue", f"£{df_filtered['Revenue'].sum():,.0f}", help="Total pendapatan kotor")
    with k2: st.metric("Active Customers", f"{df_filtered['Customer ID'].nunique():,}", help="Jumlah customer unik")
    with k3: st.metric("Total Transactions", f"{df_filtered.shape[0]:,}", help="Jumlah invoice")
    with k4: st.metric("Unique Products", f"{df_filtered['Description'].nunique():,}", help="Varian produk")

    with st.expander("📂 Lihat Sampel Data Transaksi"):
        st.dataframe(df_filtered.head(), use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📅 Revenue Trend")
        revenue_trend = df_filtered.groupby("YearMonth")["Revenue"].sum()
        if not revenue_trend.empty: st.line_chart(revenue_trend, color="#29b5e8")
        else: st.warning("Data kosong.")

    with c2:
        st.subheader("👥 Cluster Distribution")
        cluster_dist = df_filtered.groupby("Customer ID")["Cluster"].first().value_counts().sort_index()
        cluster_dist.index = [f"{i}: {CLUSTER_PROFILE[i]['name'].split('(')[0]}" for i in cluster_dist.index]
        if not cluster_dist.empty: st.bar_chart(cluster_dist, color="#ffaa00")
        else: st.warning("Data kosong.")

    st.markdown("---")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🏆 Top 10 Best Sellers")
        st.caption("Produk terlaris (Quantity)")
        top_products = df_filtered.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_products, horizontal=True)

    with c4:
        st.subheader("📍 Customer Value Map (RFM)")
        st.caption("Peta Sebaran: Recency vs Monetary")
        active_customers = df_filtered["Customer ID"].unique()
        rfm_filtered = rfm[rfm["Customer ID"].isin(active_customers)].copy()
        if not rfm_filtered.empty:
            rfm_filtered["Cluster Group"] = rfm_filtered["Cluster"].map(lambda x: CLUSTER_PROFILE.get(x, {}).get('name', str(x)))
            st.scatter_chart(rfm_filtered, x="Recency", y="Monetary", color="Cluster Group", size="Frequency", height=350)

    st.markdown("---")

    # Update nama di header tabel
    st.subheader("💎 Top 10 High Value Customers (Champions / VIP)")
    top_cust = df_filtered.groupby("Customer ID")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    top_cust = top_cust.merge(rfm[['Customer ID', 'Cluster']], on='Customer ID', how='left')
    top_cust['Cluster Group'] = top_cust['Cluster'].map(lambda x: CLUSTER_PROFILE.get(x, {}).get('name', str(x)))
    top_cust["Revenue"] = top_cust["Revenue"].apply(lambda x: f"£{x:,.0f}")
    top_cust['Customer ID'] = top_cust['Customer ID'].astype(str)
    
    st.dataframe(top_cust[['Customer ID', 'Cluster Group', 'Revenue']], use_container_width=True, hide_index=True)

    st.markdown("---")
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Filtered CSV", data=csv, file_name='filtered_transactions.csv', mime='text/csv', type="primary")

# 2. Customer Recommendation
elif menu == "Customer Recommendation":
    st.header("🎯 Customer 360° & Recommendation")
    st.caption("Profil detail customer, status kesehatan, dan rekomendasi produk personal.")

    with st.expander("💡 Belum punya ID? Klik untuk lihat Contoh Customer (Top 3 per Cluster)"):
        st.write("Salin ID di bawah ini untuk mencoba demo:")
        cols_cheat = st.columns(4)
        sorted_clusters = sorted(rfm['Cluster'].unique())
        for i, cls_id in enumerate(sorted_clusters):
            with cols_cheat[i]:
                c_prof = CLUSTER_PROFILE.get(cls_id, {})
                c_name_short = c_prof.get('name', str(cls_id)).split("(")[0]
                c_icon = c_prof.get('icon', "")
                st.markdown(f"{c_icon} {c_name_short}")
                top3_cheat = rfm[rfm['Cluster'] == cls_id].sort_values('Monetary', ascending=False).head(3)
                for _, row in top3_cheat.iterrows():
                    st.code(f"{int(row['Customer ID'])}", language="text")
                    st.caption(f"Rev: £{row['Monetary']:,.0f}")

    col_input, col_btn = st.columns([3, 1])
    with col_input: customer_id_input = st.number_input("Masukkan Customer ID", min_value=1, step=1)
    with col_btn:
        st.write("##") 
        check_btn = st.button("🔍 Analisis Customer", type="primary")

    def display_product_cards(product_list, df_source, promo_label=None):
        if not product_list or len(product_list) == 0:
            st.warning("⚠️ Data personal belum cukup. Menampilkan **Global Best Sellers** sebagai alternatif:")
            fallback_items = df_source.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(5).index.tolist()
            product_list = fallback_items
            promo_label = "🔥 Global Hot Item" 

        subset = df_source[df_source['Description'].isin(product_list)].copy()
        if subset.empty:
            st.error("Gagal memuat detail produk.")
            return

        if 'UnitPrice' not in subset.columns:
            subset = subset[subset['Quantity'] > 0] 
            subset['UnitPrice'] = subset['Revenue'] / subset['Quantity']

        stats = subset.groupby('Description').agg({'UnitPrice': 'mean', 'Quantity': 'sum'}).reset_index()
        stats = stats.set_index('Description').reindex(product_list).reset_index()
        cols = st.columns(2) 
        
        for idx, row in stats.iterrows():
            if pd.isna(row['UnitPrice']): continue
            col_target = cols[idx % 2]
            with col_target:
                with st.container(border=True):
                    if promo_label: st.markdown(f":red-background[**{promo_label}**]")
                    c_icon, c_text = st.columns([1, 4])
                    with c_icon: st.write("# 🛒")
                    with c_text:
                        st.markdown(f"**{row['Description']}**")
                        st.markdown(f"Harga: **£{row['UnitPrice']:,.2f}**")
                        st.caption(f"Terjual: {int(row['Quantity']):,} unit")

    if check_btn:
        with st.spinner("Sedang memuat model rekomendasi..."):
            topN_cluster, user_item_matrix, item_similarity_df = load_recommendation_models()
        
        results, error = recommend_products(customer_id_input, rfm, topN_cluster, user_item_matrix, item_similarity_df)

        if error: st.error(error)
        else:
            cid = normalize_customer_id_input(customer_id_input)
            cluster = results["Cluster"]
            c_profile = CLUSTER_PROFILE.get(cluster, {})
            c_name = c_profile.get("name", f"Cluster {cluster}")
            c_icon = c_profile.get("icon", "📦")
            
            rec_df = rfm.loc[rfm["Customer ID"] == cid]
            my_rec, my_freq, my_mon = rec_df["Recency"].values[0], rec_df["Frequency"].values[0], rec_df["Monetary"].values[0]

            if my_rec <= 30: status, color = "Active & Healthy 🟢", "green"
            elif my_rec <= 90: status, color = "Warning 🟡", "orange"
            else: status, color = "Churn 🔴", "red"

            st.markdown("---")
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.title(c_icon)
                    st.markdown(f"### {c_name}")
                    st.caption(f"Customer ID: {cid} | Status: :{color}[{status}]")
                with c2:
                    max_rec = rfm['Recency'].max()
                    max_freq = rfm['Frequency'].quantile(0.95) 
                    max_mon = rfm['Monetary'].quantile(0.95)
                    def calc_prog(v, m): return min((v/m), 1.0)
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Recency", f"{int(my_rec)} Hari")
                        st.progress(calc_prog(max_rec - my_rec, max_rec), "Keaktifan")
                    with m2:
                        st.metric("Frequency", f"{int(my_freq)}x")
                        st.progress(calc_prog(my_freq, max_freq), "Frekuensi")
                    with m3:
                        st.metric("Monetary", f"£{my_mon:,.0f}")
                        st.progress(calc_prog(my_mon, max_mon), "Nilai")

            st.write("### 📊 Riwayat Belanja")
            my_hist = df_full[df_full['Customer ID'] == cid].copy()
            if not my_hist.empty:
                chart_data = my_hist.groupby('YearMonth')['Revenue'].sum()
                st.area_chart(chart_data, color="#4CAF50")
            else: st.info("Tidak ada riwayat transaksi detail.")

            st.markdown("---")
            with st.container(border=True):
                c_strat_icon, c_strat_text = st.columns([0.5, 4])
                with c_strat_icon: st.write("## 📢")
                with c_strat_text:
                    st.subheader("Recommended Action")
                    promo_text = ""
                    if cluster == 1: 
                        st.success("🎯 **Strategy: VIP Treatment**")
                        st.write("Berikan akses *Early Bird* atau *Exclusive Packaging*. Jangan berikan diskon murah.")
                        promo_text = "💎 VIP EXCLUSIVE"
                    elif cluster == 3: 
                        st.error("🎯 **Strategy: Win-Back (Diskon Besar)**")
                        st.write("Berikan diskon **20-30%** atau *Free Shipping* untuk memancing transaksi.")
                        promo_text = "🏷️ DISKON 20% OFF"
                    elif cluster == 0: 
                        st.warning("🎯 **Strategy: Volume Diskon**")
                        st.write("Tawarkan 'Beli 2 Lebih Murah' atau Voucher Ongkir.")
                        promo_text = "⚡ BEST VALUE DEAL"
                    else: 
                        st.info("🎯 **Strategy: Upselling**")
                        st.write("Tawarkan produk pelengkap (Bundling) untuk menaikkan nilai belanja.")
                        promo_text = "✨ REKOMENDASI"

            st.write("## 📦 Katalog Rekomendasi")
            st.caption(f"Produk yang dikurasi khusus untuk **{c_name}** sesuai strategi di atas.")

            tab1, tab2, tab3 = st.tabs(["🔥 Top Picks (Cluster)", "🤝 You Might Like (Personal)", "🆕 Try Something New"])
            with tab1:
                st.caption(f"Barang wajib punya untuk grup {c_name}.")
                display_product_cards(results["Top Cluster Products"], df_full, promo_label=promo_text)
            with tab2:
                st.caption("Berdasarkan analisa kemiripan belanja user ini (Collaborative Filtering).")
                display_product_cards(results["Similar Products (CF)"], df_full, promo_label="❤️ FOR YOU")
            with tab3:
                st.caption("Barang populer yang belum pernah dibeli (Potensi Upsell).")
                display_product_cards(results["Cluster Products Not Bought"], df_full, promo_label="🆕 NEW ARRIVAL")

# 3. Cluster Insight
elif menu == "Cluster Insight":
    st.header("🔎 Cluster Strategic Insight")
    st.caption("Analisis mendalam perilaku segmen menggunakan data global.")

    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        cluster_options = sorted(df_full["Cluster"].unique())
        format_func = lambda x: f"{x} - {CLUSTER_PROFILE[x]['name']}"
        selected_cluster = st.selectbox("Pilih Segmen:", cluster_options, format_func=format_func)
    
    cluster_df = df_full[df_full["Cluster"] == selected_cluster]
    cluster_rfm = rfm[rfm["Cluster"] == selected_cluster]
    
    global_rec, global_freq, global_mon = rfm["Recency"].mean(), rfm["Frequency"].mean(), rfm["Monetary"].mean()
    avg_rec, avg_freq, avg_mon = cluster_rfm["Recency"].mean(), cluster_rfm["Frequency"].mean(), cluster_rfm["Monetary"].mean()
    c_profile = CLUSTER_PROFILE.get(selected_cluster, {})

    with col_info:
        with st.container(border=True):
            st.info(f"""
            ### {c_profile['icon']} {c_profile['name']}
            **Profil:** {c_profile['desc']}
            - **Populasi:** {cluster_rfm.shape[0]:,} User
            - **Total Revenue:** £{cluster_df['Revenue'].sum():,.0f}
            """)

    st.subheader("📊 Perbandingan vs Rata-rata Toko")
    c1, c2, c3 = st.columns(3)
    c1.metric("Recency", f"{avg_rec:.1f} Hari", f"{avg_rec - global_rec:.1f} vs Global", delta_color="inverse")
    c2.metric("Frequency", f"{avg_freq:.1f} Trx", f"{avg_freq - global_freq:.1f} vs Global")
    c3.metric("Monetary", f"£{avg_mon:,.0f}", f"£{avg_mon - global_mon:,.0f} vs Global")

    st.markdown("---")
    col_chart, col_prod = st.columns([2, 1])
    with col_chart:
        st.subheader("📈 Tren Revenue Segmen Ini")
        trend_cluster = cluster_df.groupby("YearMonth")["Revenue"].sum()
        if not trend_cluster.empty: st.area_chart(trend_cluster, color="#3b8ed0")
        else: st.warning("No Data")
    with col_prod:
        st.subheader("🏆 Produk Favorit")
        st.table(cluster_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(5))

    st.subheader("🚀 Rekomendasi Strategis")
    strat_col1, strat_col2 = st.columns(2)
    with strat_col1:
        with st.container(border=True):
            st.write("#### ✅ What to Do (Action)")
            if selected_cluster == 1:
                st.success("Target: **Retention & Experience**")
                st.write("- Layanan CS Prioritas.")
                st.write("- Akses Pre-order Exclusive.")
                st.write("- Fokus pada Value bukan Diskon.")
            elif selected_cluster == 3:
                st.error("Target: **Win-Back**")
                st.write("- Diskon Agresif (We Miss You).")
                st.write("- FOMO Marketing.")
            elif selected_cluster == 0:
                st.warning("Target: **Increase Basket Size**")
                st.write("- Voucher min. belanja.")
                st.write("- Cross-sell barang murah.")
            else:
                st.info("Target: **Upselling**")
                st.write("- Dorong ke produk margin tinggi.")
                st.write("- Paket Bundling Premium.")

    with strat_col2:
        with st.container(border=True):
            st.write("#### ❌ What to Avoid")
            if selected_cluster == 1:
                st.write("⛔ Jangan spam notifikasi.")
                st.write("⛔ Jangan tawarkan barang murah.")
            elif selected_cluster == 3:
                st.write("⛔ Jangan diamkan > 30 hari lagi.")
            elif selected_cluster == 0:
                st.write("⛔ Jangan hapus promo diskon.")
            else:
                st.write("⛔ Jangan abaikan potensi mereka.")
