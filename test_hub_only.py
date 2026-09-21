import sys
print("Script started", file=sys.stderr)

from hub.sync_hub import ZeroTrustSyncHub, AgentAction
print("Hub imported", file=sys.stderr)

from dv.verilog_runner import SAMPLE_COUNTER, run_dv_full
print("DV runner imported", file=sys.stderr)

# Create the Hub directly (no FastAPI)
hub = ZeroTrustSyncHub(allowed_actions=["VERIFY_HARDWARE_RTL"])
print("Hub instantiated", file=sys.stderr)

# Create an action
action = AgentAction(
    action_type='VERIFY_HARDWARE_RTL',
    payload={'verilog_code': SAMPLE_COUNTER},
    agent_claimed_belief={}
)
print("Action created", file=sys.stderr)

# Define a simple callback that calls the DV runner
def dv_callback(payload):
    print("Inside callback", file=sys.stderr)
    code = payload.get('verilog_code')
    return run_dv_full(code, None)

# Run verification
print("Calling verify_and_commit...", file=sys.stderr)
result = hub.verify_and_commit(action, execution_callback=dv_callback, state_key='TEST_DV')
print("Done", file=sys.stderr)

print('Verdict:', result.verdict)
print('Reason:', result.reason)
print('Verified State:', result.verified_state)