from ray.rllib.agents.callbacks import DefaultCallbacks
import numpy as np
import logging
import math


class attack_stats(DefaultCallbacks):

    def on_episode_start(self, worker, base_env,
                         episode, **kwargs):
        episode.user_data["success"] = 0
        episode.user_data["steps"] = 0
        episode.user_data["lp_dist"] = 0
        episode.user_data["lp_success"] = 0
        return

    def on_episode_step(self, worker, base_env,
                        episode, **kwargs):
        info = episode.last_info_for()
        if info["done"] == True:
            #print(info)
            episode.user_data["success"] = info["success"]
            episode.user_data["lp_success"] = info["lp_success"]
            episode.user_data["steps"] = info["n_step"]
            episode.user_data["lp_dist"] = info["lp_dist"]
        return

    def on_episode_end(self, worker, base_env,
                       episode, **kwargs):
        episode.custom_metrics["success"] = episode.user_data["success"]
        episode.custom_metrics["lp_success"] = episode.user_data["lp_success"]
        episode.custom_metrics["steps"] = episode.user_data["steps"]
        episode.custom_metrics["lp_dist"] = episode.user_data["lp_dist"]

        return