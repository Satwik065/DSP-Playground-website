from main import hub
from hub.sync_hub import AgentAction
from dv.verilog_runner import SAMPLE_COUNTER

# Create a test action
action = AgentAction(
    action_type='VERIFY_HARDWARE_RTL',
    payload={'verilog_code': SAMPLE_COUNTER},
    agent_claimed_belief={}
)

# Run it through the Hub
result = hub.verify_and_commit(action, state_key='TEST_DV')

print('Verdict:', result.verdict)
print('Reason:', result.reason)
print('Verified State:', result.verified_state)