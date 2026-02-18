import math
from llm_spice.execute.executor import AFExecutor, Executor
from llm_spice.model import Model
from llm_spice.passes.af_transfer import AFTransferPass
from llm_spice.utils.common import (
    Tensor,
    WorkloadStats,
    OpRunStats,
)
from llm_spice.utils.experiments import mp_tqdm

DEFAULT_FAST_MODE = True


def run_prefill(
    model: Model,
    input_tokens: int,
    num_users: int,
    executor: Executor,
):
    inp = Tensor(shape=(num_users, input_tokens, model.hf_config.hidden_size))
    if isinstance(executor, AFExecutor):
        AFTransferPass(
            executor.get_processor()[0].pcfg, executor.get_processor()[1].pcfg
        ).apply(model)
    _ = model(inp)
    prefill_stats = executor.run_model(model)
    model.forward_pass_done()
    return prefill_stats


def run_decode(
    model: Model,
    output_tokens: int,
    num_users: int,
    executor: Executor,
    fast_mode: bool = DEFAULT_FAST_MODE,
):
    if isinstance(executor, AFExecutor):
        attn_pcfg = executor.get_processor()[0].pcfg
        ffn_pcfg = executor.get_processor()[1].pcfg
        attn_microbatch = attn_pcfg.pp_size * attn_pcfg.ep_size
        ffn_microbatch = ffn_pcfg.pp_size * ffn_pcfg.ep_size
        assert num_users >= attn_microbatch, (
            f"Num users {num_users} must be greater than or equal to attn_microbatch {attn_microbatch}"
        )
        assert num_users >= ffn_microbatch, (
            f"Num users {num_users} must be greater than or equal to ffn_microbatch {ffn_microbatch}"
        )
        AFTransferPass(attn_pcfg, ffn_pcfg).apply(model)

    if fast_mode:
        kvcache = model.get_kvcache()
        assert kvcache is not None
        seq_len = model.get_seq_len()
        inp = Tensor(shape=(num_users, 1, model.hf_config.hidden_size))
        _ = model(inp)
        first_decode_stats = executor.run_model(model)
        model.forward_pass_done()
        model.insert_kvcache(num_users, seq_len + output_tokens, kvcache.dtype)
        inp = Tensor.new_like(inp)
        _ = model(inp)
        last_decode_stats = executor.run_model(model)
        model.forward_pass_done()

        assert first_decode_stats.extra_info is not None
        assert last_decode_stats.extra_info is not None

        final_attn_stats = OpRunStats.attn_interpolate(
            first_decode_stats.extra_info["attn_stats"],
            last_decode_stats.extra_info["attn_stats"],
            seq_len,
            output_tokens,
        )
        final_ffn_stats = OpRunStats.linear_interpolate(
            first_decode_stats.extra_info["ffn_stats"],
            last_decode_stats.extra_info["ffn_stats"],
            output_tokens,
        )

        final_decode_stats = final_attn_stats.merge(final_ffn_stats)
        final_decode_stats.extra_info = {
            "attn_stats": final_attn_stats,
            "ffn_stats": final_ffn_stats,
        }

        return final_decode_stats

    decode_stats = None
    for _ in mp_tqdm(range(output_tokens)):
        # Decode: one token per user per step -> (batch, seq=1, vocab)
        inp = Tensor(shape=(num_users, 1, model.hf_config.hidden_size))
        _ = model(inp)
        curr_decode_stats = executor.run_model(model)
        if decode_stats is None:
            decode_stats = curr_decode_stats
        else:
            decode_stats = decode_stats.merge(curr_decode_stats)
            if decode_stats.extra_info and "attn_stats" in decode_stats.extra_info:
                assert curr_decode_stats.extra_info is not None
                decode_stats.extra_info["attn_stats"] = decode_stats.extra_info[
                    "attn_stats"
                ].merge(curr_decode_stats.extra_info["attn_stats"])
                decode_stats.extra_info["ffn_stats"] = decode_stats.extra_info[
                    "ffn_stats"
                ].merge(curr_decode_stats.extra_info["ffn_stats"])
        model.forward_pass_done()
    assert decode_stats is not None
    return decode_stats


def run_workload(
    model: Model,
    input_tokens: int,
    output_tokens: int,
    num_users: int,
    executor: Executor,
    fast_mode: bool = DEFAULT_FAST_MODE,
):
    model.clear_kvcache()
    # Initalialization
    num_chips = executor.get_processor()[0].pcfg.num_chips
    assert num_users >= executor.get_processor()[0].pcfg.dp_size
    num_users_per_node = math.ceil(
        num_users
        / (
            executor.get_processor()[0].pcfg.dp_size
            * executor.get_processor()[0].pcfg.pp_size
        )
    )

    # Prefill: model input shaped as (batch=num_users, seq=input_tokens, vocab)
    prefill_stats = run_prefill(model, input_tokens, num_users_per_node, executor)

    # Decode
    decode_stats = run_decode(
        model, output_tokens, num_users_per_node, executor, fast_mode
    )

    throughput = (
        num_users
        * (input_tokens + output_tokens)
        / (prefill_stats.duration + decode_stats.duration)
    )

    total_tco = sum(p.get_tco(p.pcfg.num_chips) for p in executor.get_processor())

    cost_per_1m_tokens = total_tco / (throughput * 3600 / 1e6)

    return WorkloadStats(
        prefill_stats=prefill_stats,
        decode_stats=decode_stats,
        num_users=num_users,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_chips=num_chips,
        ttft=prefill_stats.duration,
        tpot=decode_stats.duration / output_tokens,
        tps_per_user=output_tokens / decode_stats.duration,
        throughput=throughput,
        throughput_per_chip=throughput / num_chips,
        ttc=prefill_stats.duration + decode_stats.duration,
        total_tco=total_tco,
        cost_per_1m_tokens=cost_per_1m_tokens,
    )


def run_workload_pd(
    model: Model,
    input_tokens: int,
    output_tokens: int,
    num_users: int,
    prefill_executor: Executor,
    decode_executor: Executor,
    fast_mode: bool = DEFAULT_FAST_MODE,
):
    model.clear_kvcache()
    prefill_pcfg = prefill_executor.get_processor()[0].pcfg
    decode_pcfg = decode_executor.get_processor()[0].pcfg
    num_chips = prefill_pcfg.num_chips + decode_pcfg.num_chips

    # Prefill: model input shaped as (batch=num_users, seq=input_tokens, vocab)
    assert num_users >= prefill_pcfg.dp_size * prefill_pcfg.pp_size
    prefill_num_users_per_node = math.ceil(num_users / prefill_pcfg.num_tp_shards)
    prefill_stats = run_prefill(
        model, input_tokens, prefill_num_users_per_node, prefill_executor
    )

    kvcache = model.get_kvcache()
    assert kvcache is not None
    # TODO: Transfer kvcache to decode chip

    # Decode
    decode_num_users_per_node = math.ceil(num_users / decode_pcfg.num_tp_shards)
    model.insert_kvcache(decode_num_users_per_node, input_tokens, kvcache.dtype)
    decode_stats = run_decode(
        model, output_tokens, decode_num_users_per_node, decode_executor, fast_mode
    )

    total_duration = max(prefill_stats.duration, decode_stats.duration)

    throughput = num_users * (input_tokens + output_tokens) / total_duration

    total_tco = sum(
        p.get_tco(p.pcfg.num_chips)
        for p in prefill_executor.get_processor() + decode_executor.get_processor()
    )

    cost_per_1m_tokens = total_tco / (throughput * 3600 / 1e6)

    return WorkloadStats(
        prefill_stats=prefill_stats,
        decode_stats=decode_stats,
        num_users=num_users,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_chips=num_chips,
        ttft=prefill_stats.duration,
        tpot=decode_stats.duration / output_tokens,
        tps_per_user=output_tokens / decode_stats.duration,
        throughput=throughput,
        throughput_per_chip=throughput / num_chips,
        total_tco=total_tco,
        cost_per_1m_tokens=cost_per_1m_tokens,
        ttc=prefill_stats.duration + decode_stats.duration,
    )


def run_workload_paf(
    model: Model,
    input_tokens: int,
    output_tokens: int,
    num_users: int,
    prefill_executor: Executor,
    af_executor: AFExecutor,
    fast_mode: bool = DEFAULT_FAST_MODE,
):
    model.clear_kvcache()
    prefill_pcfg = prefill_executor.get_processor()[0].pcfg
    attn_pcfg = af_executor.get_processor()[0].pcfg
    ffn_pcfg = af_executor.get_processor()[1].pcfg

    num_chips = prefill_pcfg.num_chips + attn_pcfg.num_chips + ffn_pcfg.num_chips

    # Prefill: model input shaped as (batch=num_users, seq=input_tokens, vocab)
    assert num_users >= prefill_pcfg.num_tp_shards
    prefill_num_users_per_node = math.ceil(num_users / prefill_pcfg.num_tp_shards)
    prefill_stats = run_prefill(
        model, input_tokens, prefill_num_users_per_node, prefill_executor
    )

    kvcache = model.get_kvcache()
    assert kvcache is not None

    # TODO: Transfer kvcache to decode chip
    # Decode
    decode_num_users_per_node = math.ceil(num_users / ffn_pcfg.num_tp_shards)
    assert num_users >= attn_pcfg.num_tp_shards
    assert num_users >= ffn_pcfg.num_tp_shards
    model.insert_kvcache(decode_num_users_per_node, input_tokens, kvcache.dtype)
    decode_stats = run_decode(
        model, output_tokens, decode_num_users_per_node, af_executor, fast_mode
    )

    assert decode_stats.extra_info is not None
    attn_stats = decode_stats.extra_info["attn_stats"]
    ffn_stats = decode_stats.extra_info["ffn_stats"]
    assert isinstance(attn_stats, OpRunStats) and isinstance(ffn_stats, OpRunStats)

    decode_duration = max(attn_stats.duration, ffn_stats.duration)
    total_duration = max(prefill_stats.duration, decode_duration)

    throughput = num_users * (input_tokens + output_tokens) / total_duration

    total_tco = sum(
        p.get_tco(p.pcfg.num_chips)
        for p in prefill_executor.get_processor() + af_executor.get_processor()
    )

    cost_per_1m_tokens = total_tco / (throughput * 3600 / 1e6)

    return WorkloadStats(
        prefill_stats=prefill_stats,
        decode_stats=decode_stats,
        num_users=num_users,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_chips=num_chips,
        ttft=prefill_stats.duration,
        tpot=decode_stats.duration / output_tokens,
        tps_per_user=output_tokens / decode_stats.duration,
        throughput=throughput,
        throughput_per_chip=throughput / num_chips,
        ttc=prefill_stats.duration + decode_stats.duration,
        total_tco=total_tco,
        cost_per_1m_tokens=cost_per_1m_tokens,
    )
