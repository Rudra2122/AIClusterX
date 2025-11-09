from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class SubmitJob(BaseModel):
    workload: str = Field(description="matmul | conv | sleep | torch_cnn | torch_ddp_mock")
    size: int = Field(ge=1, le=8192, description="problem size (or batch size for torch)")
    iterations: int = Field(ge=1, le=5000, description="iteration/steps")
    priority: str = Field(pattern="^(high|med|low)$", default="med")
    deadline_sec: Optional[int] = Field(default=None, description="deadline SLO in seconds")

class JobStatus(BaseModel):
    state: str
    worker: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    info: Optional[Dict[str, Any]] = None
    latency_sec: Optional[float] = None
    slo_violation: Optional[bool] = None
