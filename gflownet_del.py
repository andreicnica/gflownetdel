import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import random
from copy import deepcopy
from typing import List
from scipy import stats
from argparse import Namespace
from utils_del import OfflineBufferWithBest, Transition, convert_nested_dict, run_stats


# %% Environment for generating trajectories from blocks
# TODO this should be adapted to DELsData -
#  will be a bit tricky with variable actions based on state, but this design might work
#  We could use available unique block idxs (_block_space)
#  and send available blocks in state representation in order for the Model to know what
#  action "Head" to use
class DELsEnvDummy:
    def __init__(self):
        self.horizon = 3  # number of steps
        self._block_space = list(np.arange(10))
        self.observation_space = self.horizon
        self.action_space = len(self._block_space)
        # 0 - for empty block, 1 for next block, > 1 (block idx + 2)
        self._crt_state = np.zeros(self.horizon, dtype=np.int32)
        self._step = 0

    def state(self) -> dict:
        """
            Return observation and action space (list of actions available for this state) + other
            Could be useful when you will have dynamic action space based on state
            parent_count is 1 given that we work on a Tree of states
            Step in sequence construction 0 - start state ; 3 - leaf for 3 step env
        """
        # TODO return current state representation & available blocks & steps & parent_count
        state_repr = {"obs": self._crt_state.copy(), "av_blocks": self.available_blocks(),
                      "step": self._step, "parent_count": 1}
        return state_repr

    def available_blocks(self, state=None):
        # TODO Return available blocks for state or crt_state
        state = self._crt_state if state is None else state
        return list(self._block_space)

    def reset(self):
        # TODO Return state 0 representation
        self._crt_state.fill(0)
        self._crt_state[0] = 1  # highlight block to fill
        self._step = 0
        return self.state()

    def step(self, action):
        """
            Add chosen block, based on indexing by action in the list of available blocks for current state.
        """
        # TODO Implement dynamics of changing to next state
        self._crt_state[self._step] = self.available_blocks()[action] + 2
        self._step += 1

        if self._step < self.horizon:
            self._crt_state[self._step] = 1
            reward, done = 0, False
        else:
            reward = self.reward(self.state())
            done = True

        info = dict()
        return self.state(), reward, done, info

    def reward(self, state) -> float:
        # TODO return state reward (either from proxy, or database)
        # Dummy reward based on unique blocks
        return float(len(np.unique(state["obs"])))

    @staticmethod
    def unique_id(state):
        # TODO Return unique hashable id for state (this works for numpy max uint8)
        return state["obs"].astype(np.uint8).tobytes()

    def backward_sample(self, tr: Transition) -> List[Transition]:
        """Generate  trajectory going backward from a terminal transition """
        # TODO adapt for DELsData
        next_state = deepcopy(tr.next_obs)
        traj = []
        for step in range(0, self.horizon)[::-1]:
            prev_state = deepcopy(next_state)
            prev_state["obs"][step] = 1
            prev_state["step"] = step
            if step + 1 < self.horizon:
                prev_state["obs"][step + 1] = 0
            removed_block = next_state["obs"][step] - 2

            prev_state["av_blocks"] = self.available_blocks(prev_state)
            act = prev_state["av_blocks"].index(removed_block)
            if step == self.horizon - 1:
                rdone, rreward, rinfo = True, tr.reward, tr.info
            else:
                rdone, rreward, rinfo = False, 0, {}
            traj.append(Transition(prev_state, act, rreward, rdone, rinfo, next_state))
            next_state = prev_state

        return traj[::-1]


def test_env():
    """ Dummy test env """
    env = DELsEnvDummy()
    obs = env.reset()
    done = False
    traj = []
    while not done:
        act = np.random.randint(len(obs["av_blocks"]))
        next_obs, r, done, info = env.step(act)
        traj.append([obs, act, r, done, info, next_obs])
        obs = next_obs
    print("Sampled")
    for x in traj: print(x)
    print(f"LAST BLOCK STATE: {obs['obs'] - 2}")
    print("Backward sample")
    for x in env.backward_traj(obs, r, info): print(x)


# %% Models
# TODO For DELsData and different action spaces architectures might be important
#  Different model blocks based on number of blocks, different heads based on action blocks ...
#  Current design of env might help. In the state representation you also have states.av_blocks
#  which can help us choose each "action" head for each state.
#  Will need to implement defined methods which should operate on variable sizes of logits for
#  each state in a batch
class DemoModel(nn.Module):
    def __init__(self, args, obs_space, act_space, floatX, h_size=128):
        super().__init__()
        self.floatX = floatX
        self.base = nn.Sequential(
            nn.Linear(obs_space, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(h_size, act_space)
        self.value = nn.Linear(h_size, 1)

    def forward(self, states):
        """
            Should return a single value (scalar) ber state and logits per state
            value: BATCH_SIZE X 1
            logits: BATCH_SIZE X ACT_SIZE(states[i])
        """
        # TODO adapt for DELsData
        x = states.obs.to(self.floatX)
        x = self.base(x)
        logits = self.actor(x)
        value = self.value(x)
        return value, logits

    def sample_act(self, states):
        """ Return an action for each state. -> shape: BATCH_SIZE (type int64) """
        # TODO
        values, logits = self(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def log_probs(self, states, logits, actions):
        """ Get log probabilities for each action given logits from corresponding states. """
        # TODO
        dist = torch.distributions.Categorical(logits)
        return dist.log_prob(actions)

    def index_output_by_action(self, states, logits, actions):
        """ get logits[ix, actions[ix]] - need in case of variable logit sizes/state """
        # TODO
        return logits.gather(1, actions[:, None])

    def sum_output(self, states, logits):
        """ [sum(logits[ix]) for ix in range(len(state))"""
        # TODO
        return logits.sum(1)


# %% Models
class Dataset:
    """
        Used for generating (train/eval) samples from model.
        Used for keeping buffers and some logs of samples.
        Used for generating train batches (possibility to include offline backward samples)
    """
    def __init__(self, args, model, env_class: DELsEnvDummy,  s2b_transform, device, floatX):
        num_envs = getattr(args, 'num_envs_procs', 8)
        seed = getattr(args, 'seed', 142857)
        self.device = device
        self.floatX = floatX
        self.np_rng = np.random.RandomState(seed)
        self.rand_rng = random.Random(seed)
        self.mdps = [env_class() for _ in range(num_envs)]
        self.mdp = self.mdps[0]
        self._device = device
        self.model = model
        self.s2b = s2b_transform
        self.floatX = floatX

        self.offline_buffer = OfflineBufferWithBest(
            full_history_size=getattr(args, 'full_history_size', 10000),
            best_history_size=getattr(args, 'best_history_size', 10000),
            env=self.mdp
        )

        # Fraction of train batch to sample new trajectories from model vs offline backward samples
        self.train_batch_new_f = getattr(args, 'train_batch_new_f', 1.)
        # For train batch we need to sample random a bit - to always cover the state space
        self.random_action_prob = getattr(args, 'random_action_prob', 0.05)

        self.new_sampled_traj_count = 0
        self.offline_sampled_traj_count = 0
        self.train_traj_count = 0
        log_window = 1000
        self.train_sampled_r = []
        self.exclude_leaves = set()

    @torch.no_grad()
    def get_samples_model(self, num_traj, eps=0) -> List[List[Transition]]:
        """
            Get new batch of trajectories from the model. Using eps-greedy policy.
            Return list of list of transitions.
        """
        # TODO Could parallelize better
        envs = self.mdps
        model = self.model

        env_traj = [list() for _ in range(len(envs))]
        trajs = []
        obss = [xenv.reset() for xenv in envs]
        while len(trajs) < num_traj:
            bobss = self.s2b(obss, self.device)
            act = model.sample_act(bobss)

            next_obss = []
            for ix, env in enumerate(envs):
                arnd = self.np_rng.rand() > eps
                send_act = act[ix].item() if arnd else np.random.randint(len(obss[ix]["av_blocks"]))
                next_obs, r, done, info = env.step(send_act)
                env_traj[ix].append(Transition(obss[ix], act[ix], r, done, info, next_obs))

                if done:
                    trajs.append(env_traj[ix])
                    env_traj[ix] = list()
                    next_obs = env.reset()

                next_obss.append(next_obs)
            obss = next_obss

        trajs = self.rand_rng.sample(trajs, k=num_traj)
        return trajs

    def add_excluded_transitions(self, trs: List[Transition]):
        self.exclude_leaves.update([self.mdp.unique_id(tr.next_obs) for tr in trs])

    def exclude_traj(self, traj):
        """ Check if we need to exclude this trajectory. (e.g. because of test set leaves)"""
        return True if self.mdp.unique_id(traj[-1].next_obs) in self.exclude_leaves else False

    def get_train_samples_model(self, batch_size: int, **kwargs) -> List[List[Transition]]:
        """
            Use when sampling for training.
            We will also add trajectories/transitions to our buffer.
            and check if we do not need to exclude traj from training
        """
        f_trajs = []
        while len(f_trajs) < batch_size:
            trajs = self.get_samples_model(batch_size, **kwargs)
            add_traj = [traj for traj in trajs if not self.exclude_traj(traj)]
            f_trajs.extend(add_traj)

        self.offline_buffer.add_trajs(f_trajs)
        return f_trajs

    def sample_train_batch(self, batch_size: int, new_f: float = None):
        """ Create train batch or trajectories from either new model samples or offline samples """
        if self.offline_buffer.has_for_sampling():
            # Add offline trajectories only if buffer is full enough
            train_batch_new_f = self.train_batch_new_f if new_f is None else new_f
            bnew_size = int(batch_size * train_batch_new_f)
            boffline_size = batch_size - bnew_size
        else:
            bnew_size, boffline_size = batch_size, 0

        new_trajs = self.get_train_samples_model(bnew_size, eps=self.random_action_prob) if bnew_size > 0 else []
        offline_trajs = self.offline_buffer.sample_batch(boffline_size) if boffline_size > 0 else []
        ret_traj = new_trajs + offline_trajs

        self.train_sampled_r.extend([tr[-1].reward for tr in ret_traj])
        self.new_sampled_traj_count += len(new_trajs)
        self.offline_sampled_traj_count += len(offline_trajs)
        self.train_traj_count += len(ret_traj)

        return ret_traj

    def log(self):
        _log = {
            "train_new_traj_count": self.new_sampled_traj_count,
            "train_offline_traj_count": self.offline_sampled_traj_count,
            **run_stats("train_r", np.array(self.train_sampled_r)),
            **self.offline_buffer.log()
        }
        self.train_sampled_r.clear()
        return _log


# %% Transformers: from list of env (dict) states to NNModel input batch format
# -- Output should be usable directly by the NN Model to do inference on.
def states2batch(states: List[dict], device, floatX=None) -> Namespace:
    # TODO Adapt DELs Data.  Should batch a list of states for NN model input format
    batch = dict({key: [x[key] for x in states] for key in states[0].keys()})
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            batch[k] = torch.from_numpy(v).to(device)
        else:
            batch[k] = torch.tensor(v).to(device)

    return Namespace(**batch)


# %% Transformers: from List of trajectories into training batch format (batch of transitions)
def batch2trainbatch(batch_traj: List[List[Transition]], s2b, device, floatX):
    obs, r, act, done, next_obs, trajids = [], [], [], [], [], []

    for ix, traj in enumerate(batch_traj):
        for tr in traj:
            obs.append(tr.obs)
            r.append(tr.reward)
            done.append(tr.done)
            next_obs.append(tr.next_obs)
            act.append(tr.act)
        trajids += [ix] * len(traj)

    batch = Namespace(
        obs=s2b(obs, device=device, floatX=floatX),
        reward=torch.tensor(r, device=device, dtype=floatX),
        done=torch.tensor(done, device=device, dtype=torch.bool),
        act=torch.tensor(act, device=device, dtype=torch.long),
        seq=torch.tensor(trajids, device=device, dtype=torch.long),
        next_obs=s2b(next_obs, device=device, floatX=floatX),
    )
    return batch


ENVS = {
    "DELsEnvDummy": DELsEnvDummy,
}

S2B_TRANSFORMS = {
    "states2batch": states2batch,
}

TRAINER_BATCH_TRANSFORM = {
    "batch2trainbatch": batch2trainbatch,
}


# %% GFN Trainer - train step and loss construction methods.
class GFNTrainer:
    def __init__(self, args: Namespace, model, dataset, s2b, device, floatX):
        self.model = model
        self.s2b = s2b
        self.device, self.floatX = device, floatX
        self.b2b = TRAINER_BATCH_TRANSFORM[args.trainer_batch_transform]
        self.dataset = dataset
        self.args = args
        self.train_batch_size = args.mbsize

        self.opt = getattr(torch.optim, args.optim)(model.parameters(), **vars(args.optim_args))

        self.log_reg_c = getattr(args, 'log_reg_c', (0.1/8)**4)
        self.target_norm = getattr(args, 'target_norm', [-8.6, 1.10])
        self.reward_exp = getattr(args, 'reward_exp', 10)
        self.reward_norm = getattr(args, 'reward_norm', 8)
        self.R_min = getattr(args, 'R_min', 1e-8)

        self.training_traj = 0
        self.training_steps = 0

    def train_iter(self):
        """ Do 1 train step """
        tbatch = self.dataset.sample_train_batch(self.train_batch_size)
        self.training_traj += len(tbatch)

        tbatch = self.b2b(tbatch, self.s2b, self.device, self.floatX)
        tbatch.reward[tbatch.done] = self.r2r(tbatch.reward[tbatch.done])

        loss, info = self.loss(tbatch)

        self.opt.zero_grad()
        loss.backward()

        if self.args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)

        self.opt.step()
        self.training_steps += 1

        info.update({
            "loss": loss.item(),
            "training_steps": self.training_steps,
            "training_traj": self.training_traj,
            **run_stats("train_r_shaped", tbatch.reward[tbatch.done]),
            **self.dataset.log()
        })
        return info

    def loss(self, batch):
        """ Compute loss for batch of transitions """

        model = self.model
        args = self.args
        log_reg_c = self.log_reg_c

        donef = batch.done.to(self.floatX)

        # parents of the state outputs
        mol_out_obs, stem_out_obs = model(batch.obs)
        mol_out_nobs, stem_out_nobs = model(batch.next_obs)

        # index parents by their corresponding actions
        qsa_p = model.index_output_by_action(batch.obs, stem_out_obs, batch.act)

        # then sum the parents' contribution, this is the inflow - 1 parent for tree
        inflow = torch.log(torch.exp(qsa_p) + log_reg_c)

        # sum the state's Q(s,a), this is the outflow
        # TODO Useless compute for terminal states
        exp_outflow = model.sum_output(batch.next_obs, torch.exp(stem_out_nobs))

        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(log_reg_c + batch.reward + exp_outflow * (1-donef))

        losses = _losses = (inflow - outflow_plus_r).pow(2)

        term_loss = (losses * donef).sum() / (donef.sum() + 1e-20)
        flow_loss = (losses * (1-donef)).sum() / ((1-donef).sum() + 1e-20)

        if args.balanced_loss:
            # Help to focus more on having the correct loss on terminal state value prediction
            loss = term_loss * args.leaf_coef + flow_loss
        else:
            loss = losses.mean()

        info = {
            **run_stats("term_loss", term_loss.data.cpu().numpy()),
            **run_stats("flow_loss", flow_loss.data.cpu().numpy()),
            **run_stats("losses", losses.data.cpu().numpy()),
        }
        return loss, info

    def r2r(self, scores):
        """
            This is very important. Influences a lot training & final sampling score distribution.
        """
        # Offset by 4 std dev, 99.9% of the population (assuming normalized correct)
        normscore = (scores - self.target_norm[0]) / self.target_norm[1] + 4
        normscore = torch.clip(normscore, self.R_min, None)
        return (normscore / self.reward_norm) ** self.reward_exp

    def eval_trajs(self, trajs):
        """ Evaluate batch of trajectories by computing spearman corr of traj log prob and reward"""
        tbatch = self.b2b(trajs, self.s2b, self.device, self.floatX)
        with torch.no_grad():
            mol_out_obs, stem_out_obs = self.model(tbatch.obs)
            actlogprob = self.model.log_probs(tbatch, stem_out_obs, tbatch.act)
        seq_log_prob = torch.zeros(len(trajs), device=self.device, dtype=self.floatX)
        seq_log_prob.index_add_(0, tbatch.seq, actlogprob)
        rewards = [tr[-1].reward for tr in trajs]
        evals = stats.spearmanr(seq_log_prob.cpu().numpy(), rewards)
        return {"s_corr": evals.correlation, "s_pvalue": evals.pvalue}


def main(args):
    device = torch.device('cuda') if args.use_cuda and torch.cuda.is_available() else torch.device("cpu")
    floatX = getattr(torch, args.floatX)
    s2b = S2B_TRANSFORMS[args.s2b_transform]
    env_class = ENVS[args.env]
    env = env_class()  # type: DELsEnvDummy
    model = DemoModel(args.model, env.observation_space, env.action_space, floatX)
    model.to(device)
    dataset = Dataset(args.dataset, model, env_class, s2b, device, floatX)
    trainer = GFNTrainer(args.trainer, model, dataset, s2b, device, floatX=floatX)

    # ==============================================================================================
    # -- offline load
    # TODO Add backward trajectories from offline dataset. Example
    #  Easy way would be to add them to best history of states and make sure best_history_size
    #  arg is higher than dataset size. Opt 2 would be to implement a different
    #  OfflineBufferWithBest class that has fixed offline buffer from where GFNTrainer can sample
    #  offline. We had OfflineBufferWithBest to keep the best states discovered so far and
    #  sample backward from them.
    #  Transition(prev_state, act, reward, done, info, next_state)
    #  Make sure the next_state representation is in the same format as the env
    dataset_transitions = [
        Transition(
            None, None, 3, True, {},
            {"obs": np.array([4, 5, 6]), "av_blocks": env.available_blocks(), "step": 4, "parent_count": 1}
        ),
        Transition(
            None, None, 2, True, {},
            {"obs": np.array([4, 5, 4]), "av_blocks": env.available_blocks(), "step": 4, "parent_count": 1}
        ),
    ]
    assert len(dataset_transitions) < args.dataset.best_history_size, "history buffer too small"
    dataset.offline_buffer.best_history.extend([tr.reward for tr in dataset_transitions], dataset_transitions)
    eval_traj = [env.backward_sample(tr) for tr in dataset_transitions]
    dataset.add_excluded_transitions(dataset_transitions)
    # ==============================================================================================

    for train_step in range(args.train_steps):
        logs = trainer.train_iter()
        # print(f"[Train {train_step}] {logs}")

        # TODO we should log somewhere all logs. Other metrics should be computed.
        #  E.g. TopKDiverse score - to keep track of best scoring diverse (~modes) set of K states
        #  discovered.
        print(f"[Train {train_step}] s_r {logs['samples_r_mean']:.2f} "
              f"| new_r {logs['new_samples_r_mean']:.2f} "
              f"| r_shapeM {logs['train_r_shaped_max']:.2f} "
              f"| r_shapem {logs['train_r_shaped_min']:.2f} "
              f"| new {logs['new_samples_mean']:.2f}")

        if train_step % args.eval_freq == 0:
            # TODO We could also sample without random actions from model to check true sampling
            #  distribution
            # Here we can evaluate our models on a held out test set. E.g.: Compute the Spearman
            #  correlation of log probabilities of sampling test set states and their reward
            eval_log = trainer.eval_trajs(eval_traj)
            print(f"[EVAL {train_step}] "
                  f"s_corr: {eval_log['s_corr']:.4f} | s_pvalue: {eval_log['s_pvalue']:.4f}")


good_config = {
    'floatX': "float32",  # better with higher precision -> float64
    'use_cuda': True,
    's2b_transform': "states2batch",  # Transform list of states - to a batch format # TODO
    'env': "DELsEnvDummy",  # Environment # TODO
    'train_steps': 10000,  # training steps
    'eval_freq': 1,  # training steps
    "model": {},
    "dataset": {
        "seed": 142857,
        "num_envs_procs": 8,
        "full_history_size": 10000,  # Trajectories to keep in buffer
        "best_history_size": 10000,  # Terminal Transitions to keep in buffer (highest scoring ones)
        'train_batch_new_f': 0.5,
        'random_action_prob': 0.05,
    },
    "trainer": {
        "trainer_batch_transform": "batch2trainbatch",
        "mbsize": 16,

        # This args are used to make train reward roughly between 0 and 2.
        # TODO reward shaping for GflowNet training is very important!
        'target_norm': [1.5, 1.10],
        # you can play with this, higher is more risky but will give
        # higher rewards on average if it succeeds.
        'reward_exp': 10,
        'reward_norm': 6,
        'R_min': 1e-8,

        "optim": "Adam",
        "optim_args": {
            "lr": 5e-4,
            'betas': (0.9, 0.999),  # Optimization seems very sensitive to this, default value works
            "eps": 1.e-8,
        },
        'leaf_coef': 10,  # Can be much bigger, not sure what the trade off is exactly though
        'clip_grad': 0,
        'log_reg_c': 2.4414062500000004e-08,  # (0.1 / 8) ** 4,  #
        "balanced_loss": True,
    }
}


if __name__ == '__main__':
    main(convert_nested_dict(good_config))
