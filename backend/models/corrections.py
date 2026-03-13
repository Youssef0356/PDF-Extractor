"""CorrectionRecord model for the feedback loop (V2).

Stores per-field corrections from the user to enable continuous learning.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime


class CorrectionRecord(BaseModel):
    """One correction entry — a field where the user disagreed with the AI."""

    pdf_id: str = Field(..., description="Unique ID of the processed PDF")
    field_name: str = Field(..., description="Schema field name that was corrected")
    ai_extracted_value: Any = Field(None, description="Value the AI extracted (may be null)")
    user_corrected_value: Any = Field(None, description="Value the user provided")
    category: Optional[str] = Field(None, description="Instrument category at time of correction")
    typeMesure: Optional[str] = Field(None, description="typeMesure at time of correction")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    accepted: bool = Field(
        False,
        description="True if user confirmed AI value without change",
    )
    rule: Optional[str] = Field(
        None,
        description="Optional user-provided rule/explanation for the correction",
    )


class CorrectionBatch(BaseModel):
    """Payload for the POST /corrections endpoint."""

    pdf_id: str
    category: Optional[str] = None
    typeMesure: Optional[str] = None
    corrections: list[CorrectionRecord] = []
