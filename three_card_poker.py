from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from three_card_poker_env import ThreeCardPokerEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

env = normalize(ThreeCardPokerEnv())
policy = GaussianGRUPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()