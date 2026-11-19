"""Keep complete FM histories on one learner; preserve original loss mass."""
from collections import OrderedDict
import re

import torch

from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions


def trajectory_rank_groups(option_groups, world):
    from action_head_rl.worker import record_input_length

    if not option_groups or world < 1:
        raise ValueError('need FM conditions and a positive learner count')
    trajectories = OrderedDict()
    seen = set()
    for group in option_groups:
        key = group[0]['condition_id']
        match = re.fullmatch(r'(step=\d+/rollout=\d+)/option=.+', key)
        if match is None or key in seen:
            raise ValueError('FM prefix scheduling needs unique rollout/option identities')
        seen.add(key)
        trajectories.setdefault(match.group(1), []).append(group)
    # Preserve each trajectory's original order, including any legitimate
    # cache invalidation boundary. A different trajectory gets a fresh cache.
    chains = list(trajectories.values())
    # Ranks run every condition forward in lockstep, so the update takes one
    # wave per condition of the busiest rank. Measured B512 steps cost ~10 s
    # per wave independent of prefix tokens (FSDP gathers dominate), so balance
    # the condition count; token workload only orders equal-length chains.
    workloads = calculate_workload(torch.tensor([
        max(record_input_length(g[0]) for g in chain) for chain in chains],
        dtype=torch.int64)).tolist()
    costs = [len(chain) for chain in chains]
    order = sorted(range(len(chains)), key=lambda i: (costs[i], workloads[i]), reverse=True)
    chains, costs = [chains[i] for i in order], [int(costs[i]) for i in order]
    partitions = get_seqlen_balanced_partitions(costs, min(world, len(chains)), equal_size=False)
    ranks = [[group for i in partition for group in chains[i]] for partition in partitions]
    ranks.extend([] for _ in range(world - len(ranks)))
    width = max(map(len, ranks))
    template = min(option_groups, key=lambda g: (record_input_length(g[0]), len(g)))[0]
    # All ranks execute the same number of prefix + query forwards. Empty
    # slots carry zero validity; the original global denominator is untouched.
    for rank, groups in enumerate(ranks):
        while len(groups) < width:
            groups.append([dict(template,
                condition_id=f"{template['condition_id']}/prefix-padding={rank}:{len(groups)}",
                sample_valid=0., advantage=0.)])
    return ranks
