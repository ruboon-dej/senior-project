import logging
import re
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from data_split import load_split, thai_raw, gloss_clean


# ============================================================
# CONFIGURATION
# ============================================================

log = logging.getLogger("llm_gloss")

# llama.cpp OpenAI-compatible API
LLAMA_URL = "http://127.0.0.1:9931/v1/chat/completions"
LLM_MODEL = "local-model"

# Context configuration
CONTEXT_LIMIT = 39424
MAX_TOKENS = 1024

# Retrieval configuration
TOP_K = 12

# Request timeout (seconds)
REQUEST_TIMEOUT = 300


# ============================================================
# LOAD DATASET
# ============================================================

_train, _val, _test = load_split()

if not _train:
    raise ValueError("Training dataset is empty. Cannot initialize retrieval.")

# Retrieval pool: TRAIN ONLY
# Prevents validation/test examples from leaking into retrieval.
_src = [thai_raw(p[0]) for p in _train]
_tgt = [gloss_clean(p[1]) for p in _train]

# Closed vocabulary: TRAIN ONLY
# Alternatively, use a predefined gloss inventory if available.
VOCAB = sorted({
    gloss
    for pair in _train
    for gloss in gloss_clean(pair[1]).split()
})

VOCAB_SET = set(VOCAB)

if not VOCAB:
    raise ValueError("Gloss vocabulary is empty. Check gloss_clean().")


# ============================================================
# TF-IDF RETRIEVAL
# ============================================================

_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 3)
)

_mat = _vec.fit_transform(_src)


def _examples(sentence, k=TOP_K):
    """
    Retrieve the most similar Thai sentences from the training set
    using character-level TF-IDF similarity.

    Returns:
        List of (Thai sentence, ThSL gloss sequence) tuples.
    """

    sentence = thai_raw(sentence)

    if not sentence.strip():
        return []

    similarities = linear_kernel(
        _vec.transform([sentence]),
        _mat
    )[0]

    # Retrieve top-k examples
    indices = similarities.argsort()[::-1][:k]

    return [
        (_src[i], _tgt[i])
        for i in indices
    ]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_TEXT = (
    "You are a Thai Sign Language (ThSL) gloss translation system.\n"
    "Your task is to convert Thai sentences into ThSL gloss sequences.\n\n"

    "STRICT RULES:\n"

    "1. Output ONLY ThSL glosses separated by single spaces.\n"
    "2. Do NOT provide explanations, comments, or translations.\n"
    "3. Use ONLY glosses from the provided vocabulary.\n"
    "4. Spell every gloss exactly as it appears in the vocabulary.\n"
    "5. Preserve numbered variants such as แขก(2).\n"
    "6. Follow ThSL grammatical ordering rather than automatically "
    "copying Thai word order.\n"
    "7. Follow the translation patterns demonstrated in the examples.\n"
    "8. Omit particles, tense markers, and other words without "
    "corresponding glosses when appropriate.\n"
    "9. Do not invent new glosses.\n"
    "10. If a Thai word has no appropriate gloss, omit it.\n\n"

    "VOCABULARY:\n"
    + " | ".join(VOCAB)
)


# ============================================================
# LLAMA.CPP API COMMUNICATION
# ============================================================

def _chat(system_text, user_text):
    """
    Send a request to the local llama.cpp server.

    Uses the OpenAI-compatible Chat Completions API.
    """

    try:
        response = requests.post(
            LLAMA_URL,
            json={
                "model": LLM_MODEL,

                "messages": [
                    {
                        "role": "system",
                        "content": system_text
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ],

                "temperature": 0,
                "max_tokens": MAX_TOKENS,
                "stream": False
            },
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        content = data["choices"][0]["message"]["content"]

        if not content:
            raise ValueError("LLM returned an empty response.")

        # Monitor prompt token usage when available
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)

        if prompt_tokens >= CONTEXT_LIMIT - 1024:
            log.warning(
                "Prompt approaching context limit: %d/%d tokens.",
                prompt_tokens,
                CONTEXT_LIMIT
            )

        return content

    except requests.exceptions.ConnectionError:
        log.error(
            "Cannot connect to llama.cpp. "
            "Make sure the server is running at %s",
            LLAMA_URL
        )
        raise

    except requests.exceptions.Timeout:
        log.error("LLM request timed out after %d seconds.", REQUEST_TIMEOUT)
        raise

    except requests.exceptions.HTTPError:
        log.error(
            "llama.cpp returned HTTP error: %s",
            response.text[:1000]
        )
        raise

    except (KeyError, IndexError, ValueError) as error:
        log.error("Invalid LLM response: %s", error)
        raise


# ============================================================
# OUTPUT PROCESSING
# ============================================================

def _parse_glosses(response):
    """
    Clean the LLM response and extract gloss tokens.

    Removes common formatting artifacts before validation.
    """

    # Remove reasoning tags if they appear in the response
    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL
    )

    # Remove Markdown code fences
    response = response.replace("```", "")

    # Remove common labels
    response = re.sub(
        r"^\s*(Gloss|ThSL|Output)\s*:\s*",
        "",
        response,
        flags=re.IGNORECASE
    )

    return response.strip().split()


# ============================================================
# MAIN THAI -> THSL GLOSS FUNCTION
# ============================================================

def llm_glosses(sentence, retries=1):
    """
    Translate a Thai sentence into a ThSL gloss sequence.

    Pipeline:
        1. Normalize Thai input.
        2. Retrieve similar training examples using TF-IDF.
        3. Construct a RAG prompt.
        4. Generate glosses using local Qwen.
        5. Validate output against the closed vocabulary.
        6. Retry if invalid glosses are generated.

    Args:
        sentence: Thai input sentence.
        retries: Number of additional attempts if validation fails.

    Returns:
        List of valid ThSL gloss tokens.
    """

    sentence = thai_raw(sentence)

    if not sentence or not sentence.strip():
        log.warning("Received empty Thai sentence.")
        return []

    # --------------------------------------------------------
    # 1. RETRIEVE SIMILAR EXAMPLES
    # --------------------------------------------------------

    examples = _examples(sentence)

    shots = "\n".join(
        f"Thai: {thai}\nGloss: {gloss}"
        for thai, gloss in examples
    )

    # --------------------------------------------------------
    # 2. CONSTRUCT PROMPT
    # --------------------------------------------------------

    base_prompt = (
        "Here are examples of Thai sentences and their "
        "corresponding ThSL gloss sequences:\n\n"

        f"{shots}\n\n"

        "Now translate the following sentence.\n\n"

        f"Thai: {sentence}\n"
        "Gloss:"
    )

    # --------------------------------------------------------
    # 3. GENERATE AND VALIDATE
    # --------------------------------------------------------

    output = []

    for attempt in range(retries + 1):

        prompt = base_prompt

        # Provide corrective feedback on subsequent attempts
        if attempt > 0:
            prompt += (
                "\n\nYour previous response contained invalid glosses.\n"
                f"Previous response: {' '.join(output)}\n"
                f"Invalid glosses: {invalid_glosses}\n\n"
                "Please correct your response.\n"
                "Use ONLY glosses from the provided vocabulary.\n"
                "Output ONLY the corrected gloss sequence.\n"
                "Gloss:"
            )

        # Generate response
        response = _chat(SYSTEM_TEXT, prompt)

        output = _parse_glosses(response)

        # Check vocabulary constraints
        invalid_glosses = [
            gloss
            for gloss in output
            if gloss not in VOCAB_SET
        ]

        # ----------------------------------------------------
        # 4. VALID OUTPUT
        # ----------------------------------------------------

        if output and not invalid_glosses:

            log.info(
                "Successfully generated %d glosses on attempt %d.",
                len(output),
                attempt + 1
            )

            return output

        # ----------------------------------------------------
        # 5. INVALID OUTPUT
        # ----------------------------------------------------

        if not output:
            log.warning(
                "Attempt %d: LLM returned no glosses.",
                attempt + 1
            )

        else:
            log.warning(
                "Attempt %d: Invalid glosses detected: %s",
                attempt + 1,
                invalid_glosses
            )

    # --------------------------------------------------------
    # 6. FINAL FALLBACK
    # --------------------------------------------------------

    log.error(
        "Unable to generate a completely valid gloss sequence "
        "after %d attempts.",
        retries + 1
    )

    # Return only valid glosses rather than introducing
    # unknown tokens into the downstream pose pipeline.
    return [
        gloss
        for gloss in output
        if gloss in VOCAB_SET
    ]