from rllab.algos.vpg import VPG
from rllab.baselines.zero_baseline import ZeroBaseline
from three_card_poker_env import ThreeCardPokerEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy  import CategoricalMLPPolicy

env = normalize(ThreeCardPokerEnv())
policy = CategoricalMLPPolicy(env_spec=env.spec)
baseline = ZeroBaseline(env_spec=env.spec)
algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()