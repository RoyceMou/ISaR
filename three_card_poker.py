from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from three_card_poker_env import ThreeCardPokerEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.uniform_control_policy import UniformControlPolicy

env = normalize(ThreeCardPokerEnv())
policy = UniformControlPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()