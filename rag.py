from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gemini-3.8-flash"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def build_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )


SYSTEM_PROMPT = """Ти — експертний асистент із цивільного захисту та безпеки життєдіяльності.
Твоє завдання — надавати чіткі, практичні та перевірені відповіді виключно на основі наданих матеріалів ДСНС України.

ПРАВИЛА ТА ОБМЕЖЕННЯ:
1. Джерело фактів:
- Відповідай виключно на основі тексту в блоці <context>.
- Категорично заборонено додумувати деталі, екстраполювати або використовувати власні зовнішні знання, яких немає в документі.

2. Повнота та часткова відповідь:
- Якщо в контексті є повна відповідь — надай її вичерпно, уникаючи сухої відповіді одним реченням.
- Якщо в контексті є лише частина відповіді — виклади все, що відомо з тексту, і прямо зазнач, яка частина запиту не покрита матеріалами.
- Якщо в контексті взагалі немає відповіді — напиши виключно: "У наданих документах немає інформації з цього питання."

3. Структура та форматування:
- Для послідовностей дій, евакуації чи правил поведінки в небезпеці завжди використовуй нумеровані списки (1, 2, 3) за пріоритетом критичності.
- Описові пояснення формуй короткими, зрозумілими абзацами.
- Тон — спокійний, офіційно-діловий, доступний для цивільного населення. Мова — виключно українська.

4. Екстрений пріоритет:
- Якщо питання стосується безпосередньої небезпеки, у відповіді нагадай про виклик екстрених служб (101 або 112).
"""

HUMAN_PROMPT = """<context>
{context}
</context>

ПИТАННЯ КОРИСТУВАЧА:
<question>
{question}
</question>
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def build_llm(api_key):
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=api_key,
    )


def extract_text(content):
    if isinstance(content, str):
        return content

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "".join(parts)


def ask_rag(question, retriever, api_key):
    docs = retriever.invoke(question)

    context = format_docs(docs)

    messages = prompt.invoke({
        "context": context,
        "question": question,
    })

    llm = build_llm(api_key)
    response = llm.invoke(messages)

    sources = list(dict.fromkeys(
        doc.metadata.get("section", "unknown")
        for doc in docs
    ))

    return extract_text(response.content), sources