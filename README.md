# LLM SPICE

A lightweight, high-performance modeling framework for analyzing and optimizing large language model (LLM) workloads. Model execution on diverse hardware accelerators with realistic performance characteristics, evaluate disaggregation strategies, and perform TCO analysis.

## Features

- **10+ Pre-configured Models**: LLaMA 3/4, DeepSeek R1, Qwen3 MoE with full HuggingFace config integration
- **12+ Hardware Accelerators**: NVIDIA GPUs (H100, H200, GB200, GB300), wafer-scale systems (WSE-3, Craftwerk) with detailed cost/power models
- **Disaggregation Strategies**: Compare monolithic vs. disaggregated architectures (Prefill-Decode, Prefill-Attention-FFN)
- **Parallelism Support**: Tensor, data, pipeline, and expert parallelism with automatic allocation
- **Online Serving Simulation**: Continuous batching, dynamic scheduling, trace-driven workloads
- **TCO Analysis**: Hardware cost modeling with power, area, yield, and lifetime amortization
- **MoE Support**: First-class support for Mixture-of-Experts models
- **Performance Analysis**: Roofline analysis, MFU calculation, memory profiling, per-operator breakdowns

## Installation

### Prerequisites

- Python 3.10
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))

### Quick Setup

```bash
git clone https://github.com/davidchencsl/llm_spice.git
cd llm_spice
./setup.sh  # Installs uv, dependencies, and runs tests
```

Or manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run pytest
```

## Quick Start

```bash
# List available models and hardware
uv run cli.py list model
uv run cli.py list processor

# Run a simple workload: LLaMA 3 70B on 8x H100 GPUs
uv run cli.py run_static normal \
    --model=llama3_70b \
    --chip=H100 \
    --parallelism="(8,1,1)" \
    --input_tokens=1024 \
    --output_tokens=1024

# Try PAF disaggregation
uv run cli.py run_static paf \
    --model=deepseek_r1 \
    --prefill_chip=H100 --prefill_parallelism="(8,1,1)" \
    --attn_chip=H100 --attn_parallelism="(8,1,1)" \
    --ffn_chip=H100 --ffn_parallelism="(2,1,1)" \
    --input_tokens=1024 --output_tokens=1024
```

## CLI Reference

The CLI provides a hierarchical command structure powered by Python Fire.

### Commands

#### List Resources

```bash
# Show all models with parameter counts and architecture details
uv run cli.py list model

# Show all processors with FLOPS, memory, bandwidth, cost
uv run cli.py list processor

# Show processor cost breakdown
uv run cli.py show processor H100 --mode=tco --num_chips=32
```

#### Run Static Workloads

Run static batching workloads with fixed request sizes:

```bash
# Normal (monolithic) serving
uv run cli.py run_static normal \
    --model=llama3_70b --dtype=BF16 --chip=H100 \
    --parallelism="(8,1,1)" \
    --input_tokens=1024 --output_tokens=1024 --num_users=1

# Prefill-Decode disaggregation
uv run cli.py run_static pd \
    --model=llama3_405b \
    --prefill_chip=H100 --prefill_parallelism="(8,1,1)" \
    --decode_chip=H100 --decode_parallelism="(8,1,1)" \
    --input_tokens=512 --output_tokens=512

# Prefill-Attention-FFN disaggregation (recommended)
uv run cli.py run_static paf \
    --model=deepseek_r1 \
    --prefill_chip=H100 --prefill_parallelism="(8,1,1)" \
    --attn_chip=H100 --attn_parallelism="(8,1,1)" \
    --ffn_chip=H100 --ffn_parallelism="(1,1,1)" \
    --input_tokens=1024 --output_tokens=1024
```

#### Run Online Workloads

Simulate online serving with continuous batching and dynamic request arrival:

```bash
# Normal serving
uv run cli.py run_online normal \
    --model=llama3_70b --chip=H100 --parallelism="(8,1,1)" \
    --trace=AzureLLMChat2024 --req_rate=5.0 --max_num_reqs=100

# With PD disaggregation
uv run cli.py run_online pd \
    --model=llama3_405b \
    --prefill_chip=H100 --prefill_parallelism="(8,1,1)" \
    --decode_chip=H100 --decode_parallelism="(8,1,1)" \
    --trace=AzureLLMChat2024 --req_rate=5.0

# With PAF disaggregation
uv run cli.py run_online paf \
    --model=deepseek_r1 \
    --prefill_chip=H100 --prefill_parallelism="(8,1,1)" \
    --attn_chip=H100 --attn_parallelism="(8,1,1)" \
    --ffn_chip=H100 --ffn_parallelism="(2,1,1)" \
    --trace=AzureLLMChat2024 --req_rate=10.0
```

### Common Parameters

- `--model`: Model name (see `list model`)
- `--dtype`: Data type (`BF16`, `FP16`, `FP8`, `FP4`, `INT8`, `INT4`)
- `--chip`: Hardware processor (see `list processor`)
- `--parallelism`: Tuple `(tensor_parallel, data_parallel, pipeline_parallel)`
- `--input_tokens`, `--output_tokens`: Request size
- `--num_users`: Concurrent requests (static workloads)
- `--trace`: Trace dataset name (online workloads)
- `--req_rate`: Request arrival rate in requests/second (online workloads)
- `--max_num_reqs`: Maximum requests to simulate (online workloads)

### Example Workflows

```bash
# Compare hardware choices
uv run cli.py run_static normal --model=llama3_70b --chip=H100 --parallelism="(8,1,1)"
uv run cli.py run_static normal --model=llama3_70b --chip=H200 --parallelism="(8,1,1)"
uv run cli.py run_static normal --model=llama3_70b --chip=GB200 --parallelism="(4,1,1)"

# Test parallelism strategies
uv run cli.py run_static normal --model=llama3_405b --parallelism="(16,1,1)"  # High TP
uv run cli.py run_static normal --model=llama3_405b --parallelism="(8,2,1)"   # Mixed TP/DP
uv run cli.py run_static normal --model=llama3_405b --parallelism="(4,1,4)"   # Pipeline

# Analyze workload characteristics
uv run cli.py run_static normal --model=llama3_70b --input_tokens=128 --output_tokens=2048   # Generation-heavy
uv run cli.py run_static normal --model=llama3_70b --input_tokens=8192 --output_tokens=128   # Prefill-heavy
uv run cli.py run_static normal --model=llama3_70b --num_users=32  # Batch processing
```

## Python API

Use the framework programmatically:

```python
from llm_spice import Model, Processor, DataType, ParallelismConfig, SimpleExecutor
from llm_spice.execute.workload import run_workload

# Setup
model = Model.create("llama3_70b", dtype=DataType.BF16)
processor = Processor.create("H100")
processor.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
executor = SimpleExecutor(processor)

# Run workload
stats = run_workload(model, input_tokens=1024, output_tokens=1024, num_users=1, executor=executor)
print(f"Prefill: {stats.prefill_stats.latency:.3f}s, Decode: {stats.decode_stats.latency:.3f}s")
```

For disaggregated execution:

```python
from llm_spice.execute.executor import AFExecutor
from llm_spice.execute.workload import run_workload_paf

# Create separate processors
attn_proc = Processor.create("H100")
attn_proc.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
ffn_proc = Processor.create("H100")
ffn_proc.set_parallelism(ParallelismConfig(tp_size=2, dp_size=1, pp_size=1))

# Run PAF workload
prefill_executor = SimpleExecutor(attn_proc)
decode_executor = AFExecutor(attn_proc, ffn_proc)
stats = run_workload_paf(model, 1024, 1024, 1, prefill_executor, decode_executor)
```

## Architecture

The framework follows a layered architecture:

```
CLI & Experiments (cli.py, experiments/)
    ↓
Workload Simulation (workload.py, online_workload.py, scheduler.py)
    ↓
Execution Layer (SimpleExecutor, AFExecutor, Allocator)
    ↓
Model & Hardware Layers (Models, Operators, Processors, Memory)
    ↓
Common Utilities (Tensor, DataType, Registry, Trace, Breakdown)
```

**Key Design Principles:**

1. **Registry Pattern**: Models, processors, and traces are registered at import time for easy extension
2. **Operator DAG**: Models are directed acyclic graphs of operators (Linear, Attention, FFN, MoE)
3. **Hardware Abstraction**: Processors model compute (FLOPS), memory (capacity/bandwidth), and cost (TCO)
4. **Execution as Traversal**: Executors walk the operator DAG, computing performance statistics
5. **Composable Parallelism**: TP/DP/PP/EP configurations are first-class objects

## Project Structure

```
llm_spice/
├── llm_spice/              # Main package
│   ├── model/                 # Model definitions (llama3, llama4, deepseek, qwen3_moe)
│   ├── hardware/              # Hardware abstractions (processor, memory, technode, datacenter)
│   ├── op/                    # Operators (Linear, Attention, FFN, MoE)
│   ├── execute/               # Executors, allocators, workload runners
│   ├── passes/                # Optimization passes (af_transfer)
│   └── utils/                 # Utilities (common, registry, trace, breakdown)
├── tests/                     # Comprehensive test suite
├── experiments/               # Research experiments and figures
├── traces/                    # Real-world inference traces
├── cli.py                     # CLI entrypoint
└── pyproject.toml             # Dependencies and configuration
```

## Development

### Adding a New Model

1. Create `llm_spice/model/mymodel.py`:

```python
from llm_spice.model import Model
from llm_spice.utils.registry import register_model
from llm_spice.utils.common import DataType

@register_model("mymodel_7b", "org/mymodel-7b-hf")
class MyModel(Model):
    def __init__(self, model_id: str, dtype: DataType, hf_config=None):
        super().__init__(model_id, dtype, hf_config)
        # Define architecture using operators from llm_spice.op.operators
```

2. Import in `llm_spice/model/__init__.py`:
   ```python
   from . import mymodel  # noqa: F401
   ```

3. Use it: `uv run cli.py run_static normal --model=mymodel_7b`

### Adding a New Processor

Add to `llm_spice/hardware/simd.py`:

```python
from llm_spice.utils.registry import register_processor
from llm_spice.hardware.memory import Memory
from llm_spice.hardware.technode import TechNode

@register_processor(
    "MyChip",
    hw_flops_bf16=2000e12, hw_flops_fp8=4000e12, hw_flops_fp4=8000e12,
    hw_flops_efficiency=0.23, memory_bw_efficiency=0.5,
    memory=Memory.create("HBM3E-8H"), num_memory_banks=6,
    width=35.0, height=30.0, num_tiles=1, lifetime=10.0,
    tech_node=TechNode.create("CoWoS"),
    base_power=800.0, base_frequency=1.85, base_voltage=0.9,
    target_frequency=1.85, static_power_ratio=0.35, activity_factor=1.0,
    pod_size=1, rack_size=4, server_size=8,
)
```

### Code Quality & Testing

```bash
# Linting and formatting
uv run ruff check
uv run ruff format

# Type checking
uv run pyright

# Run tests
uv run pytest                      # Full suite
uv run pytest tests/test_models.py # Specific file
uv run pytest -m "not slow"        # Exclude slow tests
uv run pytest -v                   # Verbose output

# Add dependencies
uv add <package>                   # Production
uv add --group dev <package>       # Development
```

## Advanced Features

### Automatic Resource Allocation

The `Allocator` class automatically determines optimal parallelism configuration:

```python
from llm_spice.execute.allocator import Allocator

allocator = Allocator(
    model=model, input_tokens=1024, output_tokens=1024,
    total_budget=1000000,  # $1M budget
    alloc_mode=Allocator.Mode.TCO,
)
```

### Trace-Driven Simulation

Use real-world traces from Azure or define custom workload patterns:

```python
from llm_spice import Trace

trace = Trace.create("AzureLLMChat2024")
# Use with run_online_workload
```

### TCO Modeling

Comprehensive cost modeling includes:
- Die cost (area, yield, wafer cost)
- Power modeling (static/dynamic, voltage/frequency scaling)
- Memory costs (HBM2/3/3E, SRAM)
- Datacenter hierarchy (pod, rack, server)
- Lifetime amortization

View breakdowns: `uv run cli.py show processor H100 --mode=tco`

### Disaggregation Strategies

1. **Monolithic**: Single cluster for prefill and decode
2. **Prefill-Decode (PD)**: Separate prefill and decode clusters
3. **Prefill-Attention-FFN (PAF)**: Separate prefill, attention, and FFN clusters for independent scaling

Use `run_static paf` or `run_online paf` commands for PAF disaggregation.

## Use Cases

**Research**: Architecture exploration, hardware-software co-design, optimization studies, cost analysis

**Engineering**: Capacity planning, configuration tuning, performance debugging, what-if analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Run `uv run ruff check` and `uv run pytest`
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this framework in your research, please cite:

```bibtex
[Add citation information if applicable]
```
