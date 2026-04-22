"""
train_qlearning.py — Q-learning trainer for upgraded ElevatorEnv (Phase 0).

Usage:
    python train_qlearning.py
    python train_qlearning.py --episodes 5000 --floors 5 --lam 0.25
"""

import argparse
import pickle
import numpy as np
from collections import defaultdict

from elevator_env import ElevatorEnv


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.997):
        self.n_actions  = n_actions
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.Q          = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        current_q = self.Q[state][action]
        target    = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path="qtable.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)
        print(f"Q-table saved → {path}  ({len(self.Q)} states)")


def evaluate_greedy(agent, floors=5, max_steps=600, lam=0.25, peak_offpeak=True, episodes=20):
    env = ElevatorEnv(floors=floors, max_steps=max_steps, lam=lam, peak_offpeak=peak_offpeak)
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    rewards, served, waits = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        state = env.obs_to_state(obs)
        total = 0.0
        info = {"total_served": 0, "avg_wait": 0.0}

        for _ in range(max_steps):
            action = agent.choose_action(state)
            obs, rew, term, trunc, info = env.step(action)
            state = env.obs_to_state(obs)
            total += rew
            if term or trunc:
                break

        rewards.append(total)
        served.append(info["total_served"])
        waits.append(info["avg_wait"])

    agent.epsilon = old_eps
    env.close()
    return float(np.mean(rewards)), float(np.mean(served)), float(np.mean(waits))


def train(
    episodes=30000,
    floors=5,
    max_steps=600,
    lam=0.25,
    peak_offpeak=True,
    alpha=0.1,
    gamma=0.98,
    eps_start=1.0,
    eps_end=0.02,
    eps_decay=0.9995,
    eval_interval=1000,
    eval_episodes=20,
    best_path="qtable.pkl",
):
    env   = ElevatorEnv(floors=floors, max_steps=max_steps,
                        lam=lam, peak_offpeak=peak_offpeak)
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
    )

    ep_rewards, ep_served, ep_wait, ep_eps = [], [], [], []
    best_eval_reward = -np.inf

    for ep in range(episodes):
        obs, _  = env.reset()
        state   = env.obs_to_state(obs)
        total   = 0.0

        for _ in range(max_steps):
            action     = agent.choose_action(state)
            obs, rew, term, trunc, info = env.step(action)
            next_state = env.obs_to_state(obs)

            agent.update(state, action, rew, next_state, term or trunc)
            state  = next_state
            total += rew
            if term or trunc:
                break

        agent.decay_epsilon()
        ep_rewards.append(total)
        ep_served.append(info["total_served"])
        ep_wait.append(info["avg_wait"])
        ep_eps.append(agent.epsilon)

        if (ep + 1) % 200 == 0:
            avg_r = np.mean(ep_rewards[-200:])
            avg_s = np.mean(ep_served[-200:])
            avg_w = np.mean(ep_wait[-200:])
            print(f"Ep {ep+1:5d} | reward={avg_r:8.1f} | "
                  f"served={avg_s:.1f} | avg_wait={avg_w:.1f} | eps={agent.epsilon:.3f}")

        if (ep + 1) % eval_interval == 0:
            eval_r, eval_s, eval_w = evaluate_greedy(
                agent,
                floors=floors,
                max_steps=max_steps,
                lam=lam,
                peak_offpeak=peak_offpeak,
                episodes=eval_episodes,
            )
            print(
                f"  [Eval] ep={ep+1:5d} | reward={eval_r:8.1f} | "
                f"served={eval_s:.1f} | avg_wait={eval_w:.1f}"
            )
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                agent.save(best_path)
                print(f"  [Best] updated best checkpoint ({best_eval_reward:.1f})")

    env.close()
    return agent, np.array(ep_rewards), np.array(ep_served), np.array(ep_wait), np.array(ep_eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",     type=int,   default=30000)
    parser.add_argument("--floors",       type=int,   default=5)
    parser.add_argument("--max-steps",    type=int,   default=600)
    parser.add_argument("--lam",          type=float, default=0.25)
    parser.add_argument("--no-peak",      action="store_true")
    parser.add_argument("--alpha",        type=float, default=0.1)
    parser.add_argument("--gamma",        type=float, default=0.98)
    parser.add_argument("--eps-start",    type=float, default=1.0)
    parser.add_argument("--eps-end",      type=float, default=0.02)
    parser.add_argument("--eps-decay",    type=float, default=0.9995)
    parser.add_argument("--eval-interval",type=int,   default=1000)
    parser.add_argument("--eval-episodes",type=int,   default=20)
    parser.add_argument("--out",          type=str,   default="qtable.pkl")
    args = parser.parse_args()

    agent, rewards, served, wait, eps = train(
        episodes     = args.episodes,
        floors       = args.floors,
        max_steps    = args.max_steps,
        lam          = args.lam,
        peak_offpeak = not args.no_peak,
        alpha        = args.alpha,
        gamma        = args.gamma,
        eps_start    = args.eps_start,
        eps_end      = args.eps_end,
        eps_decay    = args.eps_decay,
        eval_interval= args.eval_interval,
        eval_episodes= args.eval_episodes,
        best_path    = args.out,
    )

    final_out = args.out.replace(".pkl", "_final.pkl")
    agent.save(final_out)
    np.savez("training_stats.npz",
             rewards=rewards, served=served,
             avg_wait=wait,   epsilons=eps)
    print(f"Done. Best checkpoint → {args.out}")
    print(f"Done. Final checkpoint → {final_out}")
    print("Done. Stats → training_stats.npz")