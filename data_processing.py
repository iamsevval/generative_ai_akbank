import pandas as pd
import plotly.express as px
import os

# Veri seti yolları
DATASETS_DIR = "datasets"
CSV_FILES = {
    "salary_basic": os.path.join(DATASETS_DIR, "Software_Professional_Salaries.csv"),
    "salary_extra": os.path.join(DATASETS_DIR, "Salary_Dataset_with_Extra_Features.csv"),
    "jobs_basic": os.path.join(DATASETS_DIR, "Software_Engineer_Salaries.csv"),
    "postings": os.path.join(DATASETS_DIR, "postings2.csv")
}

def load_data():
    """Tüm veri setlerini yükleyip bir sözlük olarak döndürür."""
    dfs = {}
    for key, path in CSV_FILES.items():
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                dfs[key] = df
            else:
                dfs[key] = pd.DataFrame()
        except Exception as e:
            print(f"Hata yüklenirken: {path} - {e}")
            dfs[key] = pd.DataFrame()
    return dfs

def get_salary_chart(df, selected_field):
    """Plotly kullanarak maaş bar grafiği oluşturur."""
    if df.empty:
        return None
        
    df_field_salary = df[df["Job Title"].str.contains(selected_field, case=False, na=False)]
    if df_field_salary.empty:
        return None
        
    avg_salary = df_field_salary.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        avg_salary, 
        x="Job Title", 
        y="Salary", 
        title=f"💰 {selected_field} Alanı Ortalama Maaşlar",
        labels={"Job Title": "Pozisyon", "Salary": "Ortalama Maaş ($)"},
        color="Salary",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig, avg_salary

def get_top_companies_chart(df, selected_field):
    """Plotly kullanarak en çok iş ilanı olan şirketlerin grafiği oluşturulur."""
    if df.empty:
        return None
        
    df_field_jobs = df[df["Job Title"].str.contains(selected_field, case=False, na=False)]
    if df_field_jobs.empty:
        return None
        
    top_companies = df_field_jobs["Company"].value_counts().head(10).reset_index()
    top_companies.columns = ["Company", "Count"]
    
    fig = px.bar(
        top_companies, 
        x="Count", 
        y="Company", 
        orientation='h',
        title=f"🏢 {selected_field} Alanında En Çok İlan Açan Şirketler",
        labels={"Company": "Şirket", "Count": "İlan Sayısı"},
        color="Count",
        color_continuous_scale="Oranges"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig, df_field_jobs[["Job Title", "Company", "Salary"]].head(10)

def get_career_paths():
    """Statik kariyer yolu önerileri döndürür."""
    return {
        "Mobile": {
            "Learning_Path": ["Kotlin veya Swift öğrenin", "Mobil UI/UX temellerini çalışın", "API ve veri yönetimini öğrenin"],
            "Projects": ["Todo uygulaması", "Hava durumu uygulaması", "Chat uygulaması"],
            "Similar_Fields": ["Web", "Full Stack", "Backend"]
        },
        "Web": {
            "Learning_Path": ["HTML, CSS, JavaScript öğrenin", "Frontend framework: React, Angular, Vue", "Backend: Node.js veya Django öğrenin"],
            "Projects": ["Portfolio sitesi", "E-ticaret sitesi", "Blog platformu"],
            "Similar_Fields": ["Frontend", "Backend", "Full Stack"]
        },
        "Backend": {
            "Learning_Path": ["Python, Java veya C# öğrenin", "API geliştirme ve veri tabanı yönetimi", "RESTful servisler oluşturun"],
            "Projects": ["Blog API", "Stok yönetim sistemi", "RESTful servisler"],
            "Similar_Fields": ["Full Stack", "DevOps", "Web"]
        },
        "Frontend": {
            "Learning_Path": ["HTML, CSS, JavaScript öğrenin", "React, Angular veya Vue öğrenin", "UI/UX ve responsive design çalışın"],
            "Projects": ["Landing page", "Dashboard", "E-ticaret frontend"],
            "Similar_Fields": ["Web", "Full Stack", "Mobile"]
        },
        "Game": {
            "Learning_Path": ["Unity veya Unreal Engine öğrenin", "C# veya C++ öğrenin", "2D/3D oyun tasarımı ve fizik motoru"],
            "Projects": ["Basit platformer", "Puzzle oyunu", "Mini RPG"],
            "Similar_Fields": ["AI", "Mobile", "Backend"]
        },
        "Full Stack": {
            "Learning_Path": ["Frontend: React veya Vue", "Backend: Node.js, Django veya Flask", "Veritabanı yönetimi ve deployment"],
            "Projects": ["E-ticaret platformu", "Sosyal ağ sitesi", "Blog platformu"],
            "Similar_Fields": ["Frontend", "Backend", "Web"]
        },
        "DevOps": {
            "Learning_Path": ["CI/CD araçlarını öğrenin", "Docker ve Kubernetes kullanın", "AWS, GCP veya Azure öğrenin"],
            "Projects": ["Otomatik deploy pipeline oluşturun"],
            "Similar_Fields": ["Backend", "Full Stack", "Cloud"]
        },
        "AI": {
            "Learning_Path": ["Python, NumPy, Pandas öğrenin", "TensorFlow veya PyTorch ile ML ve DL öğrenin", "Temel makine öğrenmesi algoritmalarını çalışın"],
            "Projects": ["Basit chatbot", "Görüntü sınıflandırma", "Tahmin modelleri"],
            "Similar_Fields": ["Data Science", "Backend", "Game"]
        }
    }
