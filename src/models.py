"""Provider factory: returns the embedding model and chat LLM for the chosen provider."""
from src import config


def get_embeddings():
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        # embed_documents uses RETRIEVAL_DOCUMENT and embed_query uses RETRIEVAL_QUERY
        # automatically, which is what we want for RAG.
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            output_dimensionality=config.EMBEDDING_DIM,
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)


def get_llm():
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Temperature intentionally left at Google's default: Gemini 3 models are
        # documented to work best at 1.0. Grounding comes from the prompt + the
        # structured `answerable` flag + the retrieval gate, not from temperature.
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_LLM_MODEL, google_api_key=config.GOOGLE_API_KEY
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.OPENAI_LLM_MODEL, temperature=0)
