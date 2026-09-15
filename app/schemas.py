from pydantic import BaseModel

class ModerationRequest(BaseModel):
    text: str

class ClassScores(BaseModel):
    normal: float
    flagged: float

class ThresholdData(BaseModel):
    threshold: float
    margin: float

class ModerationResponse(BaseModel):
    prediction: str
    scores: ClassScores
    flagged: bool
    threshold_info: ThresholdData