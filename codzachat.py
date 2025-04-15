#CODZA .env dosyasındaki API anahtarlarını yükler
from dotenv import load_dotenv

#CODZA Tavily arama aracını getirir (web search için)
from langchain_community.tools.tavily_search import TavilySearchResults

#CODZA OpenAI'den ChatGPT modelini kullanır
from langchain_openai import ChatOpenAI

#CODZA Kullanıcının mesajını tanımlar
from langchain_core.messages import HumanMessage

#CODZA LangGraph'tan hazır bir ReAct ajanı oluşturmak için fonksiyon
from langgraph.prebuilt import create_react_agent

#CODZA LangGraph için SQLite tabanlı hafıza (geçmiş konuşmalar) tutar
from langgraph.checkpoint.sqlite import SqliteSaver

#CODZA .env dosyasını yükler, API anahtarlarını ortam değişkenlerine aktarır
load_dotenv()

#CODZA GPT-4 modelini kullanmak için OpenAI üzerinden model oluşturulur
model = ChatOpenAI(model="gpt-4")

#CODZA Tavily arama aracı oluşturulur (maksimum 2 sonuç getirir)
search = TavilySearchResults(max_results=2)

#CODZA Kullanılacak tüm araçları bir listeye ekliyoruz
tools = [search]

#CODZA Araçları modele bağlıyoruz, böylece model onları kullanabilir
model_with_tools = model.bind_tools(tools)

#CODZA SQLite hafızayı context manager ile başlatır (geçici bellek - ":memory:")
config = {"configurable": {"thread_id": "abc123"}}

#CODZA Ana uygulama başlatılır
if __name__ == '__main__':
    #CODZA Belleği geçici olarak başlatıyoruz (with bloğu içinde)
    with SqliteSaver.from_conn_string(":memory:") as memory:
        #CODZA Model ve araçlarla bir "ReAct" ajanı oluşturuluyor
        agent_executor = create_react_agent(model, tools, checkpointer=memory)
        
        #CODZA Kullanıcıdan sürekli olarak giriş alınır (sonsuz döngü)
        while True:
            user_input = input(">")  # Kullanıcıdan mesaj alınır
            
            #CODZA Ajan model girdiye göre yanıt üretir ve akış halinde çıktılar verir
            for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content=user_input)]}, config
            ):
                print(chunk)  # Her bir parçayı yazdır
                print("----")  # Ayırıcı
