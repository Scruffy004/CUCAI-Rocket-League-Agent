import numpy as np
import os
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger




class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityBallToGoalReward, FaceBallReward, EventReward, AlignBallGoal, RewardIfBehindBall, LiuDistanceBallToGoalReward, LiuDistancePlayerToBallReward
    from weighted_state_setter import WeightedSampleSetter 
    from custom_rewards import JumpTouchReward, InAirReward, TouchVelReward, SpeedTowardBallReward, BallCenterReward, FlipResetReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.state_setters import RandomState, DefaultState
    from your_act import LookupAction

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))


    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = CombinedReward.from_zipped((EventReward(goal=6, demo=0.1), 50),
                            (RewardIfBehindBall(VelocityBallToGoalReward()), 1.1),
                            (FaceBallReward(), 0.08),
                            (TouchVelReward(), 1),
                            (JumpTouchReward(), 0.3),
                            (SpeedTowardBallReward(), 0.2),
                            (AlignBallGoal(), 0.4),
                            (LiuDistanceBallToGoalReward(), 0.4),
                            (LiuDistancePlayerToBallReward(), 0.3),
                            (FlipResetReward(), 100)
                          )

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    
    

    state_setter = WeightedSampleSetter([RandomState(True, True, False), DefaultState()], (0.8, 0.2))

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

   # latest_checkpoint_dir = "data/checkpoints/rlgym-ppo-run/" + str(max(os.listdir("data/checkpoints/rlgym-ppo-run"), key=lambda d: int(d)))


    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=2,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=100e20,
                      log_to_wandb=True,
                      policy_lr=2e-4,
                      critic_lr=2e-4,
                      policy_layer_sizes=(1024, 1024, 1024, 1024, 512),
                      critic_layer_sizes=(2048, 1024, 1024, 1024, 512),
                      add_unix_timestamp=False,
                      #checkpoint_load_folder=latest_checkpoint_dir,
                      device="cuda")
    learner.learn() 