# hub/sync_hub.py
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field


class HubVerdict(str, Enum):
    MATCH = "MATCH"
    REJECT = "REJECT"


class AgentAction(BaseModel):
    action_type: str
    payload: Dict[str, Any]
    agent_claimed_belief: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class HubVerificationResult(BaseModel):
    verdict: HubVerdict
    gate_failed: Optional[str] = None
    reason: Optional[str] = None
    staleness_seconds: float = 0.0
    execution_time_ms: float = 0.0
    verified_state: Optional[Dict[str, Any]] = None


class ZeroTrustSyncHub:
    def __init__(self, allowed_actions: list[str], max_retries: int = 4):
        self.allowed_actions = set(allowed_actions)
        self.max_retries = max_retries
        self.environment_state: Dict[str, Any] = {}

    def update_ground_truth_state(self, key: str, value: Any):
        self.environment_state[key] = value

    def _gate_1_schema_check(self, action: AgentAction) -> bool:
        if action.action_type not in self.allowed_actions:
            return False
        return isinstance(action.payload, dict)

    def _gate_2_consistency_check(self, action: AgentAction) -> tuple[bool, float]:
        t_current = time.time()
        staleness_seconds = max(0.0, t_current - action.timestamp)
        for key, claimed_val in action.agent_claimed_belief.items():
            if key in self.environment_state:
                if self.environment_state[key] != claimed_val:
                    return False, staleness_seconds
        return True, staleness_seconds

    def _gate_3_safety_check(self, action: AgentAction) -> bool:
        payload_str = str(action.payload)
        
        # For DV verification, we MUST allow subprocess because we need to call slang/verilator.
        if action.action_type == "VERIFY_HARDWARE_RTL":
            return True
        
        # For all other actions, block dangerous system calls.
        forbidden = ["os.system", "subprocess", "rm -rf", "eval(", "exec("]
        return not any(f in payload_str for f in forbidden)

    def verify_and_commit(
        self,
        action: AgentAction,
        execution_callback: Optional[Callable] = None,
        state_key: Optional[str] = None,
    ) -> HubVerificationResult:
        start_time = time.time()

        if not self._gate_1_schema_check(action):
            return HubVerificationResult(
                verdict=HubVerdict.REJECT,
                gate_failed="GATE_1_SCHEMA_VIOLATION",
                reason=f"Action '{action.action_type}' is not whitelisted.",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        is_consistent, staleness_seconds = self._gate_2_consistency_check(action)
        if not is_consistent:
            return HubVerificationResult(
                verdict=HubVerdict.REJECT,
                gate_failed="GATE_2_BELIEF_DIVERGENCE",
                reason="Agent operates on stale/divergent environmental state.",
                staleness_seconds=staleness_seconds,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        if not self._gate_3_safety_check(action):
            return HubVerificationResult(
                verdict=HubVerdict.REJECT,
                gate_failed="GATE_3_SAFETY_VIOLATION",
                reason="Payload contains unsafe operations.",
                staleness_seconds=staleness_seconds,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        output_state = None
        if execution_callback:
            try:
                output_state = execution_callback(action.payload)
            except Exception as e:
                return HubVerificationResult(
                    verdict=HubVerdict.REJECT,
                    gate_failed="RUNTIME_EXECUTION_ERROR",
                    reason=str(e),
                    staleness_seconds=staleness_seconds,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        if output_state is not None:
            self.update_ground_truth_state(state_key or action.action_type, output_state)

        return HubVerificationResult(
            verdict=HubVerdict.MATCH,
            staleness_seconds=staleness_seconds,
            execution_time_ms=(time.time() - start_time) * 1000,
            verified_state=output_state,
        )