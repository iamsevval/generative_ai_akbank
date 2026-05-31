import streamlit as st
from dotenv import load_dotenv
import os
from rag_engine import RAGEngine
from data_processing import load_data, get_salary_chart, get_top_companies_chart

# Ortam Ayarları
load_dotenv()

st.set_page_config(page_title="Yazılım Kariyer Danışmanı", page_icon="🤖", layout="wide")
st.title("💼 AI Yazılım Kariyer Yol Haritası & Danışmanı")

# RAG Motorunu Başlat (Sadece 1 Kere)
@st.cache_resource(show_spinner="Veritabanı hazırlanıyor, lütfen bekleyin...")
def get_rag_engine():
    engine = RAGEngine()
    engine.initialize_vector_store()
    return engine

# Verileri Yükle
@st.cache_data
def get_cached_data():
    return load_data()

try:
    if not os.getenv("GOOGLE_API_KEY") and not st.secrets.get("GOOGLE_API_KEY"):
         st.error("🚨 GOOGLE_API_KEY bulunamadı. Lütfen .env veya Streamlit Secrets'e ekleyin.")
         st.stop()
except Exception:
    if not os.getenv("GOOGLE_API_KEY"):
         st.error("🚨 GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasına ekleyin.")
         st.stop()

rag_engine = get_rag_engine()
dfs = get_cached_data()

# Tab Yapısı
tab_chat, tab_data = st.tabs(["💬 Kariyer Danışmanı (Sohbet)", "📊 Veri Analizi"])

# --- TAB 1: SOHBET ---
with tab_chat:
    st.markdown("### AI Kariyer Danışmanınız Sizi Dinliyor")
    st.info("💡 Örneğin: 'Mobile alanında hangi projeleri yapabilirim?', 'Backend developer ortalama ne kadar maaş alır?' veya 'Web geliştirme öğrenme rotası nasıldır?'")
    
    # Sohbet Geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Ben yapay zeka destekli kariyer danışmanınızım. Sana yazılım alanında nasıl yardımcı olabilirim?"}]

    # Geçmişi Göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcı Girdisi
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG ile Cevap Üret
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                # Geçmişi birleştir (Sadece son 4 mesaj)
                history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])
                answer = rag_engine.get_answer(prompt, history)
                st.markdown(answer)
                
        # Cevabı kaydet
        st.session_state.messages.append({"role": "assistant", "content": answer})

# --- TAB 2: VERİ ANALİZİ ---
with tab_data:
    st.markdown("### 📈 Sektör ve Maaş Analizleri")
    software_fields = ["Mobile", "Web", "Backend", "Frontend", "Game", "Full Stack", "DevOps", "AI"]
    selected_field = st.selectbox("İncelemek istediğiniz yazılım alanını seçin:", software_fields)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Ortalama Maaşlar")
        df_salary = dfs.get("salary_basic")
        if df_salary is not None and not df_salary.empty:
            result = get_salary_chart(df_salary, selected_field)
            if result:
                fig_salary, avg_salary_df = result
                st.plotly_chart(fig_salary, use_container_width=True)
                with st.expander("Detaylı Veriyi Gör"):
                    st.dataframe(avg_salary_df.rename(columns={"Job Title": "Pozisyon", "Salary": "Ortalama Maaş"}))
            else:
                st.warning(f"{selected_field} alanı için yeterli maaş verisi bulunamadı.")
        else:
             st.warning("Maaş veri seti yüklenemedi.")

    with col2:
        st.markdown("#### En Çok İlan Veren Şirketler")
        df_jobs = dfs.get("jobs_basic")
        if df_jobs is not None and not df_jobs.empty:
            result = get_top_companies_chart(df_jobs, selected_field)
            if result:
                fig_jobs, jobs_sample_df = result
                st.plotly_chart(fig_jobs, use_container_width=True)
                with st.expander("Örnek İş İlanlarını Gör"):
                    st.dataframe(jobs_sample_df.rename(columns={"Job Title": "Pozisyon", "Company": "Şirket", "Salary": "Beklenen Maaş"}))
            else:
                st.warning(f"{selected_field} alanı için yeterli iş ilanı verisi bulunamadı.")
        else:
             st.warning("İş ilanları veri seti yüklenemedi.")
