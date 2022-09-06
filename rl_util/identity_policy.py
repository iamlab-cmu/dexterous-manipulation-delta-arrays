class IdentityLowLevelPolicy:
    def __init__(self, dim_action_space):
        self._num_params = dim_action_space
        self.num_steps = 1  # tbd from env
        self.action_dim = dim_action_space
        self.policy_params = None

    def num_params(self):
        return self._num_params

    def update_policy(self, policy_params, context=None):
        assert len(policy_params) == self._num_params
        self.policy_params = policy_params

    def action_from_state(self, step_index, observation, context=None):
        return self.policy_params

    def reward_from_state(self, step_index, observation, context=None):
        return 0.0