#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL checkpoint on a HiveBoard task (Isaac Lab's unified entrypoint).

Also exports the policy to ``<run>/exported`` (TorchScript/ONNX) for deployment::

    uv run python scripts/rl/play.py --task Isaac-HiveBoard-Anymal-BallValve-RL-Play-v0 \\
        --checkpoint logs/rsl_rl/anymal_ball_valve_teacher/<run>/model_<it>.pt --viz newton

Use ``--agent rsl_rl_distillation_cfg_entry_point`` for a student checkpoint.
Use ``--publication --stop-on-termination`` to record only the first episode in RTX.
For success-rate statistics use ``scripts/rl/evaluate.py``.
"""

import argparse
import sys
from contextlib import ExitStack
from unittest.mock import patch

import isaaclab_hiveboard_rl  # noqa: F401  (registers the HiveBoard tasks)
from train import with_newton_default

from isaaclab_rl import run_play_cli


def run_recording(argv: list[str], *, publication=False, stop_on_termination=False, output_dir=None):
    """Configure isolated upstream hooks for video output and first-episode playback."""
    from isaaclab.envs.utils import video_recorder
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_rl.entrypoints.backends import play_rsl_rl
    from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

    original_playback = play_rsl_rl.run_playback
    original_step = video_recorder.VideoRecorder.step
    finished = False
    original_pre_launch = play_rsl_rl.pre_launch_video_config
    original_clip = video_recorder.ImageSequenceClip
    if original_clip is None:
        raise ImportError("moviepy is required for video recording")

    def configure(env_cfg, args_cli):
        if output_dir is not None or stop_on_termination:
            args_cli.num_envs = 1
        if not publication:
            original_pre_launch(env_cfg, args_cli)
            if output_dir is not None:
                env_cfg.video_recorders = [VideoRecorderCfg(output_dir=output_dir)]
            return
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.sim.visualizer_cfgs = [
            NewtonRTXVisualizerCfg(
                headless=True,
                eye=env_cfg.viewer.eye,
                lookat=env_cfg.viewer.lookat,
                rtx_environment="studio",
                render_settings={"omni:rtx:quality": ("Int", 100)},
            )
        ]
        # Keep simulation-time playback: the recorder captures once per policy step.
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_rtx", output_dir=output_dir, output_filename_prefix="publication"
            )
        ]
        original_pre_launch(env_cfg, args_cli)

    class PublicationClip(original_clip):
        def write_videofile(self, filename, **kwargs):
            # Resample to 50 FPS at encoding time without speeding up the policy rollout.
            kwargs.update(fps=50, preset="slow", ffmpeg_params=["-crf", "12"])
            return super().write_videofile(filename, **kwargs)

    class EpisodeFinished(Exception):
        pass

    def record_step(recorder):
        nonlocal finished
        # The upstream env records after autoreset. Skip that frame, just as core play does.
        if bool(recorder._env.reset_buf.any()):
            finished = True
            return
        original_step(recorder)

    def playback(step, **kwargs):
        def first_episode_step():
            step()
            if finished:
                raise EpisodeFinished

        try:
            original_playback(first_episode_step, **kwargs)
        except EpisodeFinished:
            print("[INFO] First episode terminated; finishing video.")

    with ExitStack() as stack:
        stack.enter_context(patch.object(play_rsl_rl, "pre_launch_video_config", configure))
        if publication:
            stack.enter_context(patch.object(video_recorder, "ImageSequenceClip", PublicationClip))
        if stop_on_termination:
            stack.enter_context(patch.object(video_recorder.VideoRecorder, "step", record_step))
            stack.enter_context(patch.object(play_rsl_rl, "run_playback", playback))
        return run_play_cli(["--rl_library", "rsl_rl", *argv])


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--publication", action="store_true")
    parser.add_argument("--stop-on-termination", action="store_true")
    parser.add_argument("--video-output", default=None)
    options, argv = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.publication or options.stop_on_termination or options.video_output:
        argv = [*argv, "--video"]
        if options.publication:
            argv.extend(["--viz", "newton_rtx"])
        return run_recording(
            with_newton_default(argv),
            publication=options.publication,
            stop_on_termination=options.stop_on_termination,
            output_dir=options.video_output,
        )
    return run_play_cli(["--rl_library", "rsl_rl", *with_newton_default(argv)])


if __name__ == "__main__":
    sys.exit(main() or 0)
