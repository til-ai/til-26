import json
import os

import requests
from dotenv import load_dotenv
from til_environment import bomberman_env
from til_environment.config import default_config
from tqdm import tqdm, trange

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

NUM_ROUNDS = 6
MAX_SCORE = 500


def main(novice: bool):
    config = default_config()
    config.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=config)
    # be the agent at index 0
    _agent = env.possible_agents[0]
    rewards = {agent: 0 for agent in env.possible_agents}

    for i in trange(NUM_ROUNDS):
        print("Round", i + 1)
        env.reset()
        # reset endpoint
        _ = requests.post("http://localhost:5005/ae")

        for agent in tqdm(env.agent_iter(), total=1200, leave=False):
            observation, reward, termination, truncation, info = env.last()
            observation = {
                k: v if type(v) in (int, float) else v.tolist()
                for k, v in observation.items()
            }

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                action = None
            elif agent == _agent:
                response = requests.post(
                    "http://localhost:5005/ae",
                    data=json.dumps({"instances": [{"observation": observation}]}),
                )
                predictions = response.json()["predictions"]

                action = int(predictions[0]["action"])
            else:
                # take random action from other agents
                action = env.action_space(agent).sample()
            env.step(action)
    env.close()
    print(f"total rewards: {rewards[_agent]}")
    print(f"score: {rewards[_agent] / NUM_ROUNDS / MAX_SCORE}")


if __name__ == "__main__":
    main(TEAM_TRACK == "novice")
