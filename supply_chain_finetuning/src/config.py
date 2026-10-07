"""Central configuration — every tunable value lives here (no magic numbers)."""
from __future__ import annotations

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DIVERSITY_DIR = DATA_DIR / "diversity"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
RAW_JSONL = DATA_DIR / "raw.jsonl"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
VALID_JSONL = DATA_DIR / "valid.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"
TEACHER_PROMPT_TXT = DATA_DIR / "teacher_system_prompt.txt"
BASE_GENERATIONS_JSON = OUTPUTS_DIR / "base_generations.json"
TUNED_GENERATIONS_JSON = OUTPUTS_DIR / "tuned_generations.json"
JUDGE_RESULTS_JSON = OUTPUTS_DIR / "judge_results.json"
METRICS_SUMMARY_JSON = OUTPUTS_DIR / "metrics_summary.json"
AUDIT_TEMPLATE_CSV = OUTPUTS_DIR / "manual_audit_template.csv"
TEACHER_USAGE_CSV = LOGS_DIR / "teacher_usage.csv"
JUDGE_USAGE_CSV = LOGS_DIR / "judge_usage.csv"
TEACHER_CACHE_JSON = OUTPUTS_DIR / ".teacher_cache.json"
JUDGE_CACHE_JSON = OUTPUTS_DIR / ".judge_cache.json"

def ensure_dirs() -> None:
    """Create data/output/log directories on demand (idempotent)."""
    for directory in (DATA_DIR, DIVERSITY_DIR, OUTPUTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

# Domain taxonomy
ANOMALY_CLASSES = (
    "demand_spike",
    "demand_collapse",
    "supplier_shipment_delay",
    "port_logistics_congestion",
    "weather_disruption",
    "quality_hold",
    "labor_strike",
    "freight_inflation",
    "inventory_obsolescence",
    "warehouse_capacity_breach",
)
NONE_CLASS = "none"
TAXONOMY = ANOMALY_CLASSES + (NONE_CLASS,)

# Control families produce perfectly normal feeds: is_anomaly=false, class "none".
CONTROL_FAMILIES = ("holiday_peak_normal", "benign_forecast_correction")
FAMILY_TO_CLASS = {name: name for name in ANOMALY_CLASSES}
FAMILY_TO_CLASS.update({name: NONE_CLASS for name in CONTROL_FAMILIES})
SCENARIO_FAMILIES = ANOMALY_CLASSES + CONTROL_FAMILIES  # 12 families

PRODUCT_CATEGORIES = (
    "electronics", "pharmaceuticals", "apparel", "food_and_beverage", "automotive_parts",
)
REGIONS = ("North America", "Europe", "Asia-Pacific")
SEVERITY_LEVELS = ("mild", "severe", "critical")
# Allowed final severity per scenario level; controls are always severity 1.
SEVERITY_BANDS = {"mild": (1, 2), "severe": (3, 4), "critical": (4, 5)}
CONTROL_SEVERITY = 1

# Sampling and acceptance gates
MATRIX_EXPECTED_SIZE = 540          # 12 families x 5 categories x 3 regions x 3 levels
SAMPLE_TARGET = 150                 # final accepted dataset size
SAMPLE_OVERGENERATE = 170           # tuples attempted to survive rejects (~14/family)
SEED = 42
MAX_ATTEMPTS_PER_TUPLE = 3
NEAR_DUPLICATE_THRESHOLD = 85       # rapidfuzz token_set_ratio reject threshold
MAX_FAMILY_SHARE = 0.15             # running per-family share cap (150 * 0.15 = 22.5)
INPUT_KEYS = (
    "product_category", "region", "order_volume_vs_forecast_pct",
    "inventory_days_on_hand", "on_time_delivery_pct", "supplier_status",
    "transit_days_normal", "transit_days_observed", "port_congestion_index",
    "weather_event", "labor_document_note", "unit_cost_vs_last_quarter_pct", "notes",
)

# Output contract
ROOT_CAUSE_SENTENCE_RANGE = (2, 4)
ACTIONS_COUNT_RANGE = (3, 5)
SENTENCE_SPLIT_REGEX = r"(?<=[.!?])\s+(?=[A-Z])"
CONFIDENCE_MIN, CONFIDENCE_MAX = 0.0, 1.0
SEVERITY_MIN, SEVERITY_MAX = 1, 5
RAG_CONFIDENCE_THRESHOLD = 0.55     # stretch-goal retrieval trigger (disabled by default)
ENABLE_RAG = False

# Split
SPLIT_SIZES = {"train": 120, "valid": 15, "test": 15}   # exactly 80/10/10 of 150

# Student models (fine-tuning)
STUDENT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
FALLBACK_STUDENT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # Apache-2.0, if templates fail
PHI3_TARGET_MODULES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PHI3_TARGET_MODULES_MINIMAL = ("qkv_proj", "o_proj")        # OOM fallback
FALLBACK_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MAX_SEQ_LENGTH = 1024
TOKEN_FIT_SHARE = 0.95           # >= 95% of full chat examples must fit MAX_SEQ_LENGTH

# LLM access (teacher + judge)
ENV_TEACHER_BASE_URL = "TEACHER_BASE_URL"
ENV_TEACHER_API_KEY = "TEACHER_API_KEY"
ENV_TEACHER_MODEL = "TEACHER_MODEL"
DEFAULT_TEACHER_BASE_URL = "https://api.commandcode.ai/provider/v1"
DEFAULT_TEACHER_MODEL = "z-ai/glm-5.3-flash"
ENV_JUDGE_BASE_URL = "JUDGE_BASE_URL"
ENV_JUDGE_API_KEY = "JUDGE_API_KEY"
ENV_JUDGE_MODEL = "JUDGE_MODEL"
SUGGESTED_JUDGE_MODEL = "llama-3.3-70b-versatile"   # Groq free tier (preferred judge)
TEACHER_TEMPERATURE = 0.7          # variety for synthetic data
JUDGE_TEMPERATURE = 0.0            # deterministic scoring
LLM_MAX_TOKENS = 2500              # thinking models burn reasoning tokens BEFORE the JSON; a truncated object wastes a whole attempt, so the budget must fit reasoning + full JSON
JUDGE_MAX_TOKENS = 700
DISABLE_THINKING = True             # structured generation wastes latency on reasoning tokens; gates provide quality control
LLM_MAX_ATTEMPTS = 3
LLM_BACKOFF_BASE_S = 1.5
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Ground guard: claims that must not appear without a backing input field
BLANK_FIELD_VALUES = ("", "none", "n/a", "na", "none reported", "no alerts", "-")
GROUND_GUARD_RULES = (
    {
        "field": "labor_document_note",
        "keywords": ("strike", "labor", "union", "walkout", "work stoppage"),
    },
    {
        "field": "weather_event",
        "keywords": ("storm", "hurricane", "flood", "typhoon", "snow", "drought", "wildfire"),
    },
)

# Diversity report
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "was were will with due been being than then their there these those which while "
    "who why yet after before between during into over under above below not no but "
    "because report reports shows show level levels rate ratio percent also however".split()
)
TOP_TERMS_COUNT = 25
