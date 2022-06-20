from gym.envs.registration import register

register(
    id='bb_attack-v0',
    entry_point='gym_bb_attack.envs.attack_env:Attack_Env',
)