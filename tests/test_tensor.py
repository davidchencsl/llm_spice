from llm_spice.utils.common import Tensor


def test_tensor():
    tensor = Tensor(shape=(10, 10))
    assert tensor[0].shape == (10,)
    assert tensor[0:5].shape == (5,10)