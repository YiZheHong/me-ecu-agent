from dataclasses import dataclass
from typing import Optional, List

from enum import Enum

class ModelType(str, Enum):
    BASE = "Base"
    PLUS = "Plus"


class DocStatus(str, Enum):
    ONLINE = "online"
    LEGACY = "legacy"

@dataclass(frozen=True)
class DocMeta:
    doc_uid: str
    source_filename: str

    product_line: str            # "ECU"
    series: str                  # "ECU-700", "ECU-800"
    model_type: ModelType        # "Base" | "Plus"

    covered_models: List[str]    # ["ECU-750"] / ["ECU-850b"]
    model_inherits_from: Optional[str]

    status: DocStatus            # "online" | "legacy"
