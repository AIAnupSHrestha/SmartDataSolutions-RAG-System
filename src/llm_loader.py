import logging
from langchain_ollama.llms import OllamaLLM

logger = logging.getLogger("DeepSeek-R1_RAG_API")

def load_deepseek_llm():
    """
    Load Ollama DeepSeek R1 model as LangChain-compatible LLM.
    Returns:
        Ollama: LangChain LLM wrapper for DeepSeek R1
    """
    model_name = "deepseek-r1:1.5b"

    try:
        logger.info(f"Loading model {model_name}...")

        deepseek = OllamaLLM(model=model_name)


        logger.info("Ollama DeepSeek R1 model loaded successfully.")
        return deepseek

    except Exception as e:
        logger.error(f"Failed to load Ollama DeepSeek R1 model: {e}", exc_info=True)
        raise RuntimeError(f"Error loading Ollama model: {e}")