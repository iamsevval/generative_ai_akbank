# 💼 Yazılım Kariyer Yol Haritası & Danışmanı (AI RAG Bot)

## Projenin Amacı
Bu proje, kullanıcıların ilgilendiği yazılım alanına göre:
- Ortalama maaşlar,
- Top şirketlerden iş ilanları,
- Önerilen öğrenme yolları ve projeler,
- Benzer kariyer alanları  

bilgilerini sunan **Gerçek RAG (Retrieval-Augmented Generation) tabanlı bir Yapay Zeka Kariyer Danışmanı** geliştirmeyi amaçlamaktadır. Proje kapsamında veriler LangChain ve FAISS kullanılarak vektörel olarak işlenmektedir.

---

## Veri Setleri Hakkında
Projede kullanılan veri setleri hazır veri setleridir:

| Veri Seti | Açıklama |
|-----------|----------|
| Software_Professional_Salaries.csv | Yazılım profesyonellerinin iş unvanı, şirket ve maaş bilgileri |
| Salary_Dataset_with_Extra_Features.csv | Maaş verilerine ek bilgiler (lokasyon, deneyim yılı vb.) |
| Software_Engineer_Salaries.csv | Yazılım mühendisleri için detaylı maaş verileri |
| postings2.csv | Farklı şirketlerden yazılım iş ilanları |
| career_path_in_all_field.csv | Yazılım alanındaki kariyer yollarını gösterir |
| computer_science_student_career_datasetMar62024.csv | Bilgisayar bilimi öğrencilerinin kariyer tercihleri |

---

## Kullanılan Teknolojiler ve Mimari
Proje, profesyonel standartlara uygun olarak modüler bir yapıda geliştirilmiştir:
- **Web Arayüzü:** Streamlit (`st.chat_message` bileşenleri ve Sekmeler (Tabs))
- **Veri İşleme ve Görselleştirme:** Pandas ve **Plotly** (İnteraktif grafikler için)
- **RAG Pipeline:** LangChain (RecursiveCharacterTextSplitter, StuffDocumentsChain)
- **Vektör Veritabanı:** FAISS
- **Dil Modeli & Embeddings:** OpenAI GPT-4o-mini ve OpenAI Embeddings

### Proje Modülleri
- `chatbot_project.py`: Ana arayüz ve sohbet akışını yönetir.
- `rag_engine.py`: FAISS vektör veritabanını oluşturur ve Retriever zincirini kurgular.
- `data_processing.py`: Verileri CSV'den okuyup Plotly ile görselleştirir.
- `prompts.py`: Yapay zeka sisteminin kişilik (danışman) rolünü barındırır.

---

## Canlı Uygulama

Akbank Generative AI Bootcamp kapsamında hazırlanan uygulamaya [buradan ulaşabilirsiniz](https://generative-ai-akbank.streamlit.app/).
*(Not: Yeni LangChain/FAISS entegrasyonlu versiyon lokal çalışmaya uygun olarak güncellenmiştir.)*

---

## Nasıl Çalıştırılır?

1. Kütüphaneleri kurun:
```bash
pip install -r requirements.txt
```

2. `.env` dosyasını oluşturup API anahtarınızı ekleyin:
```
OPENAI_API_KEY="sk-proj-..."
```

3. Uygulamayı başlatın:
```bash
streamlit run chatbot_project.py
```
