import re
import time
import random
from logger import logger

def parse_retry_delay(error_msg: str) -> float:
    """Extract retry delay seconds from Gemini rate limit error message if available."""
    match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.5  # Add 1.5s safety buffer
    return 0.0

def embed_documents_safe(
    embed_model,
    texts: list[str],
    batch_size: int = 15,
    delay_between_batches: float = 1.5,
    max_retries: int = 6
) -> list[list[float]]:
    """
    Embeds documents in controlled batches with rate limiting and exponential backoff retry logic
    to prevent Gemini 429 RESOURCE_EXHAUSTED / quota limit errors.
    """
    embeddings = []
    total_texts = len(texts)
    total_batches = (total_texts + batch_size - 1) // batch_size
    
    logger.info(f"Starting batch embedding for {total_texts} chunks across {total_batches} batches (batch_size={batch_size}).")

    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        retries = 0
        
        while True:
            try:
                logger.info(f"🔍 Embedding batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
                print(f"🔍 Embedding batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
                batch_embeds = embed_model.embed_documents(batch_texts)
                embeddings.extend(batch_embeds)
                time.sleep(delay_between_batches)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"❌ Exceeded max retries ({max_retries}) for batch {batch_num}. Error: {e}")
                        raise e
                    
                    delay_from_msg = parse_retry_delay(err_str)
                    if delay_from_msg > 0:
                        wait_time = delay_from_msg
                    else:
                        wait_time = (2 ** retries) * 5 + random.uniform(1.0, 3.0)
                    
                    msg = f"⚠️ Rate limit (429) hit on batch {batch_num}/{total_batches}. Waiting {wait_time:.1f}s before retry (Attempt {retries}/{max_retries})..."
                    logger.warning(msg)
                    print(msg)
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Error embedding batch {batch_num}: {e}")
                    raise e
                    
    return embeddings

def embed_query_safe(embed_model, query: str, max_retries: int = 5) -> list[float]:
    """
    Embeds a single query string with retry logic for 429 rate limit errors.
    """
    retries = 0
    while True:
        try:
            return embed_model.embed_query(query)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                retries += 1
                if retries > max_retries:
                    logger.error(f"❌ Exceeded max retries for query embedding. Error: {e}")
                    raise e
                delay_from_msg = parse_retry_delay(err_str)
                wait_time = delay_from_msg if delay_from_msg > 0 else (2 ** retries) * 3 + random.uniform(1.0, 2.0)
                msg = f"⚠️ Query embedding rate limited (429). Retrying in {wait_time:.1f}s (Attempt {retries}/{max_retries})..."
                logger.warning(msg)
                print(msg)
                time.sleep(wait_time)
            else:
                raise e
