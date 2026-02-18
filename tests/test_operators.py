from llm_spice.utils.common import DataType
from llm_spice.op.operators import ElementWiseOp, FFN, Linear, Tensor


def test_linear():
    inp = Tensor(shape=(3, 4))
    gemm = Linear(4, 5)
    _ = gemm(inp)
    print(gemm)
    assert gemm.get_total_flop() == 2 * 3 * 4 * 5

    inp = Tensor(shape=(8, 3, 4))
    gemm = Linear(4, 5)
    _ = gemm(inp)
    print(gemm)
    assert gemm.get_total_flop() == 2 * 8 * 3 * 4 * 5


def test_elementwise_op():
    inp = Tensor(shape=(3, 4))
    op = ElementWiseOp()
    _ = op(inp)
    print(op)
    assert op.get_total_flop() == 3 * 4


def test_ffn():
    inp = Tensor(shape=(1024, 2048))
    ffn = FFN(2048, 2048 * 4)
    _ = ffn(inp)
    print(ffn.pretty_str())
    print(ffn.get_total_flop())


def test_ffn_fp4():
    inp = Tensor(shape=(1024, 2048))
    ffn = FFN(2048, 2048 * 4, dtype=DataType.FP4)
    _ = ffn(inp)
    print(ffn.pretty_str())
    print(ffn.get_total_flop())
    print(ffn.get_total_weights_bytes())
    print(ffn.get_total_num_params())

    assert ffn.get_total_num_params() * ffn.dtype.itemsize == ffn.get_total_weights_bytes()

