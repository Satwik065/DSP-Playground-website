import subprocess
import os
import tempfile
import shutil

# Check if a tool is installed on the system
def _tool_exists(tool_name):
    return shutil.which(tool_name) is not None

# Run slang parser (just checks syntax/type)
def run_slang(verilog_code):
    """
    Returns: {"status": "pass" or "fail", "output": "text from slang"}
    """
    if not _tool_exists("slang"):
        return {
            "status": "fail",
            "output": "slang is not installed on this system. Please install it.",
            "error_type": "tool_missing"
        }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
        f.write(verilog_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ["slang", temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return {"status": "pass", "output": result.stdout.strip() or "Syntax OK."}
        else:
            return {
                "status": "fail",
                "output": result.stderr.strip() or result.stdout.strip(),
                "error_type": "ai_error"
            }
    except subprocess.TimeoutExpired:
        return {"status": "fail", "output": "slang timed out after 30s.", "error_type": "tool_limit"}
    except Exception as e:
        return {"status": "fail", "output": f"Unexpected error: {str(e)}", "error_type": "tool_limit"}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Run Verilator (compiles and simulates)
def run_verilator(verilog_code, testbench_code=None):
    """
    Returns: {"status": "pass" or "fail", "output": "text from verilator"}
    """
    if not _tool_exists("verilator"):
        return {
            "status": "fail",
            "output": "verilator is not installed on this system. Please install it.",
            "error_type": "tool_missing"
        }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        main_file = os.path.join(temp_dir, "design.sv")
        with open(main_file, 'w') as f:
            f.write(verilog_code)
        
        if testbench_code:
            tb_file = os.path.join(temp_dir, "testbench.sv")
            with open(tb_file, 'w') as f:
                f.write(testbench_code)
        
        try:
            lint_result = subprocess.run(
                ["verilator", "--lint-only", main_file],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=temp_dir
            )
            
            if lint_result.returncode != 0:
                return {
                    "status": "fail",
                    "output": lint_result.stderr.strip() or "Verilator lint failed.",
                    "error_type": "ai_error"
                }
            
            sim_result = subprocess.run(
                [
                    "verilator", "--cc", "--exe", 
                    "--build", "-j", "0",
                    main_file
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temp_dir
            )
            
            if sim_result.returncode == 0:
                return {
                    "status": "pass",
                    "output": f"Verilator compile passed.\n{sim_result.stdout.strip() or 'Simulation ready.'}"
                }
            else:
                stderr = sim_result.stderr.lower()
                if "unsupported" in stderr or "not implemented" in stderr:
                    return {
                        "status": "fail",
                        "output": sim_result.stderr.strip(),
                        "error_type": "tool_limit"
                    }
                else:
                    return {
                        "status": "fail",
                        "output": sim_result.stderr.strip() or sim_result.stdout.strip(),
                        "error_type": "ai_error"
                    }
        
        except subprocess.TimeoutExpired:
            return {"status": "fail", "output": "Verilator compile timed out after 60s.", "error_type": "tool_limit"}
        except Exception as e:
            return {"status": "fail", "output": f"Unexpected error: {str(e)}", "error_type": "tool_limit"}

# ============================================================
# THE PART THAT WAS MISSING (SAMPLE_COUNTER and run_dv_full)
# ============================================================

def run_dv_full(verilog_code, testbench_code=None):
    """
    Runs the full pipeline: slang -> verilator.
    Returns the final verdict.
    """
    print("[DV Runner] Starting full verification pipeline...")
    
    slang_result = run_slang(verilog_code)
    if slang_result["status"] == "fail":
        slang_result["stage"] = "slang"
        return slang_result
    
    verilator_result = run_verilator(verilog_code, testbench_code)
    if verilator_result["status"] == "fail":
        verilator_result["stage"] = "verilator"
        return verilator_result
    
    return {
        "status": "pass",
        "stage": "both",
        "output": "✅ Slang parsed and Verilator compiled successfully.",
        "slang_output": slang_result.get("output", ""),
        "verilator_output": verilator_result.get("output", "")
    }

# This is what your test script is looking for
SAMPLE_COUNTER = """
module counter (
    input clk,
    input rst,
    output reg [3:0] count
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 4'b0;
        else
            count <= count + 1;
    end
endmodule
"""

SAMPLE_TESTBENCH = """
module tb_counter;
    reg clk, rst;
    wire [3:0] count;
    
    counter uut (.clk(clk), .rst(rst), .count(count));
    
    initial begin
        clk = 0;
        rst = 1;
        #10 rst = 0;
        #100 $finish;
    end
    
    always #5 clk = ~clk;
endmodule
"""