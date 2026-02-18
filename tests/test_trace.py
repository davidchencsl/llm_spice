from llm_spice.utils import Trace
import pytest

def test_azure_trace():
    trace = Trace.create("AzureLLMCode2024")
    print(len(trace))

    count = 0
    for req in trace:
        print(req)
        count += 1
        if count >= 10:
            break
    
    trace.reset()

    for req in trace[:10]:
        print(req)
    
    print(trace.average_input_output_tokens())

    trace.close()

@pytest.mark.slow
def test_all_traces_average_input_output_tokens():
    for trace in Trace.get_all_traces():
        trace = Trace.create(trace)
        print(f"{trace.name} Avg Input/Output: {trace.average_input_output_tokens()}")

def test_synthetic_trace():
    trace = Trace.create("Normal1024:4096")
    for req in trace[:10]:
        print(req)
    print(trace.average_input_output_tokens())
    trace.close()