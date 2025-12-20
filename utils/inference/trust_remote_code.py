# utils/inference/trust_remote_code.py

"""
Allowlist for model prefixes that should have trust_remote_code=True.

These are well-known model families that require custom code execution
for proper tokenization or model architecture support.
"""

# Model name prefixes (case-insensitive) that are allowed to use trust_remote_code=True
# Major upstream publishers / major labs (official orgs on HF)
TRUST_REMOTE_CODE_PREFIXES = (
    # Big frontier / foundation model labs
    "openai/",          # official OpenAI org on HF :contentReference[oaicite:0]{index=0}
    "Anthropic/",       # Anthropic org exists on HF :contentReference[oaicite:1]{index=1}
    "google/",          # official Google org on HF :contentReference[oaicite:2]{index=2}
    "meta-llama/",      # official Llama org on HF :contentReference[oaicite:3]{index=3}
    "facebook/",        # AI at Meta org slug (facebook) :contentReference[oaicite:4]{index=4}
    "microsoft/",
    "mistralai/",       # official Mistral org :contentReference[oaicite:5]{index=5}
    "xai-org/",         # xAI org :contentReference[oaicite:6]{index=6}
    "deepseek-ai/",
    "Qwen/",

    # Other major commercial model publishers
    "Cohere/",
    "CohereLabs/",
    "CohereForAI/",
    "ai21labs/",
    "databricks/",
    "nvidia/",          # NVIDIA org :contentReference[oaicite:7]{index=7}
    "amazon/",
    "apple/",
    "ibm-granite/",
    "RedHatAI/",

    # Major research / open model labs
    "allenai/",         # Ai2 org :contentReference[oaicite:8]{index=8}
    "EleutherAI/",
    "bigcode/",
    "togethercomputer/",
    "stabilityai/",     # Stability AI org :contentReference[oaicite:9]{index=9}
    "black-forest-labs/",

    # Large Asia-based model orgs commonly used upstream
    "ByteDance-Seed/",  # ByteDance Seed org :contentReference[oaicite:10]{index=10}
    "tencent/",         # Tencent org :contentReference[oaicite:11]{index=11}
    "Tencent-Hunyuan/",
    "baidu/",
    "baichuan-inc/",
    "internlm/",
    "01-ai/",
    "tiiuae/",
    "openbmb/",
    "MiniMaxAI/",
    "stepfun-ai/",
    "Alibaba-NLP/",
    "zai-org/",         # Z.ai / ChatGLM-family org :contentReference[oaicite:12]{index=12}

    # High-signal infra / ecosystem orgs (optional)
    "nomic-ai/",
    "huggingface/",
    "mosaicml/",
    "Salesforce/",
    "unsloth/",
)



def should_trust_remote_code(model_name: str) -> bool:
    """
    Check if a model should have trust_remote_code enabled.

    Args:
        model_name: The model name/path to check

    Returns:
        True if the model matches an allowed prefix (case-insensitive)
    """
    model_lower = model_name.lower()
    for prefix in TRUST_REMOTE_CODE_PREFIXES:
        if model_lower.startswith(prefix.lower()):
            return True
    return False
