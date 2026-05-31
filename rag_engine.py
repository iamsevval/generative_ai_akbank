import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from prompts import SYSTEM_PROMPT
import pandas as pd
from data_processing import load_data, get_career_paths

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store_path = "faiss_index"
        self.vector_store = None
        self.retriever = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.chain = None
        
    def prepare_documents(self):
        """CSV verilerini metin belgelerine dönüştürür"""
        dfs = load_data()
        documents = []
        
        # Maaş verilerini birleştir
        df_salary_basic = dfs.get("salary_basic", pd.DataFrame())
        df_salary_extra = dfs.get("salary_extra", pd.DataFrame())
        salary_combined = pd.concat([df_salary_basic, df_salary_extra], ignore_index=True)
        
        if not salary_combined.empty:
            # Eksik sütunları engellemek için dict yapısı
            for _, row in salary_combined.iterrows():
                title = row.get("Job Title", "Bilinmiyor")
                company = row.get("Company Name", "Bilinmiyor")
                salary = row.get("Salary", "Bilinmiyor")
                text = f"Pozisyon: {title}. Şirket: {company}. Maaş: {salary}."
                documents.append(text)
                
        # İş ilanları verilerini birleştir
        df_jobs = dfs.get("jobs_basic", pd.DataFrame())
        df_postings = dfs.get("postings", pd.DataFrame())
        jobs_combined = pd.concat([df_jobs, df_postings], ignore_index=True)
        
        if not jobs_combined.empty:
            for _, row in jobs_combined.iterrows():
                title = row.get("Job Title", row.get("job_title", "Bilinmiyor"))
                company = row.get("Company", row.get("company", "Bilinmiyor"))
                salary = row.get("Salary", "Bilinmiyor")
                text = f"İş İlanı Pozisyonu: {title}. İlan Veren Şirket: {company}. Beklenen Maaş: {salary}."
                documents.append(text)
                
        # Kariyer Yolu Bilgilerini de Vektöre Ekle
        career_paths = get_career_paths()
        for field, data in career_paths.items():
            learning = ", ".join(data["Learning_Path"])
            projects = ", ".join(data["Projects"])
            similar = ", ".join(data["Similar_Fields"])
            text = f"{field} yazılım alanı kariyer yolu: Öğrenilmesi gerekenler; {learning}. Yapılabilecek örnek projeler; {projects}. Benzer alanlar; {similar}."
            documents.append(text)

        # Eğer çok büyükse sadece ilk 1000 satırı al (Performans için opsiyonel)
        if len(documents) > 2000:
            documents = documents[:2000]

        return documents

    def initialize_vector_store(self):
        """Vektör DB oluşturur veya var olanı yükler"""
        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            texts = self.prepare_documents()
            if not texts:
                raise ValueError("Veri setleri bulunamadı, indeks oluşturulamıyor.")
            
            # Langchain için uygun formata çevir
            from langchain.schema import Document
            docs = [Document(page_content=t) for t in texts]
            
            # Text Splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(docs)
            
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            self.vector_store.save_local(self.vector_store_path)
            
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self._setup_chain()

    def _setup_chain(self):
        """LangChain Retrieval QA chain kurar"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def get_answer(self, question, chat_history=""):
        """Soruyu yanıtlar"""
        if not self.chain:
            return "Hata: RAG sistemi başlatılamadı."
            
        response = self.chain.invoke({
            "input": question,
            "question": question,
            "chat_history": chat_history
        })
        return response["answer"]
