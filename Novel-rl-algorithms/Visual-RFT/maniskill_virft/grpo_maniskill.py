#!/usr/bin/env python3
# Copyright 2025 Modified for ManiSkill robot control tasks
# Based on Visual-RFT's GRPO implementation
#
# This file should be copied to: Visual-RFT/src/virft/src/open_r1/grpo_maniskill.py
# Then run from Visual-RFT/src/virft/ using train_grpo.sh

import os
import re
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

# Import Visual-RFT's trainer
# This will work when the file is placed in src/open_r1/
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for GRPO training on ManiSkill tasks
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy_action_mse", "format"],
        metadata={"help": "List of reward functions"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


# ============================================================================
# Action Parsing Utilities
# ============================================================================

def parse_action_from_response(response):
    """
    Parse action array from model response

    Expected formats:
    - <answer>[0.1234, -0.5678, 0.9012, 1.0000]</answer>
    - [0.1234, -0.5678, 0.9012, 1.0000]

    Returns:
        numpy.ndarray of shape (4,) or None if parsing fails
    """
    # Try to extract answer from <answer> tags
    content_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if content_match:
        answer_str = content_match.group(1).strip()
    else:
        answer_str = response.strip()

    # Remove whitespace and newlines
    answer_str = answer_str.replace('\n', '').replace(' ', '')

    # Try to parse array [x, y, z, gripper]
    try:
        match = re.search(r'\[([-\d.,\s]+)\]', answer_str)
        if match:
            values_str = match.group(1)
            values = [float(v.strip()) for v in values_str.split(',')]
            if len(values) == 4:
                return np.array(values)
    except (ValueError, AttributeError):
        pass

    return None


# ============================================================================
# Reward Functions for Robot Action Prediction
# ============================================================================

def accuracy_reward_action_mse(completions, solution, **kwargs):
    """
    Reward function based on Mean Squared Error (MSE) between predicted and ground truth actions

    Reward formula: reward = exp(-k * MSE)
    - MSE = 0 -> reward = 1.0
    - Larger MSE -> reward approaches 0

    Args:
        completions: List of model completions
        solution: List of ground truth solutions (action strings)
        **kwargs: Additional arguments

    Returns:
        List of rewards (floats in [0, 1])
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Sensitivity parameter: higher k means more sensitive to errors
    k = 5.0

    for content, sol in zip(contents, solution):
        # Parse ground truth action
        try:
            # If solution is JSON string like "[0.1, 0.2, 0.3, 1.0]"
            gt_action = np.array(json.loads(sol))
        except:
            # If solution is in <answer> format
            gt_action = parse_action_from_response(f"<answer>{sol}</answer>")

        # Parse predicted action
        pred_action = parse_action_from_response(content)

        # Compute reward
        if pred_action is None or gt_action is None:
            reward = 0.0
            mse = float('inf')
        else:
            mse = np.mean((pred_action - gt_action) ** 2)
            reward = float(np.exp(-k * mse))

        rewards.append(reward)

        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Action MSE Reward: {reward:.4f} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Predicted action: {pred_action}\n")
                f.write(f"Ground truth: {gt_action}\n")
                if pred_action is not None and gt_action is not None:
                    f.write(f"MSE: {mse:.6f}\n")
                f.write("\n")

    return rewards


def accuracy_reward_action_threshold(completions, solution, **kwargs):
    """
    Binary reward function with threshold-based accuracy

    Gives reward = 1.0 if Mean Absolute Error (MAE) < threshold, else 0.0
    This is stricter and encourages exact predictions

    Args:
        completions: List of model completions
        solution: List of ground truth solutions
        **kwargs: Additional arguments

    Returns:
        List of rewards (0.0 or 1.0)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Threshold for "correct" prediction
    threshold = 0.1  # MAE < 0.1 is considered correct

    for content, sol in zip(contents, solution):
        # Parse ground truth
        try:
            gt_action = np.array(json.loads(sol))
        except:
            gt_action = parse_action_from_response(f"<answer>{sol}</answer>")

        # Parse prediction
        pred_action = parse_action_from_response(content)

        # Compute reward
        if pred_action is None or gt_action is None:
            reward = 0.0
            mae = float('inf')
        else:
            mae = np.mean(np.abs(pred_action - gt_action))
            reward = 1.0 if mae < threshold else 0.0

        rewards.append(reward)

        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Action Threshold Reward: {reward:.4f} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Predicted action: {pred_action}\n")
                f.write(f"Ground truth: {gt_action}\n")
                if pred_action is not None and gt_action is not None:
                    f.write(f"MAE: {mae:.6f}, Threshold: {threshold}\n")
                f.write("\n")

    return rewards


def format_reward(completions, **kwargs):
    """
    Reward function that checks if the completion follows the required format

    Expected format: <think>...</think><answer>[x, y, z, gripper]</answer>

    Args:
        completions: List of model completions
        **kwargs: Additional arguments

    Returns:
        List of rewards (0.0 or 1.0)
    """
    # Pattern that matches: <think>anything</think><answer>[numbers]</answer>
    pattern = r"<think>.*?</think>\s*<answer>\s*\[[-\d.,\s]+\]\s*</answer>"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    return [1.0 if match else 0.0 for match in matches]


# Reward function registry
reward_funcs_registry = {
    "accuracy_action_mse": accuracy_reward_action_mse,
    "accuracy_action_threshold": accuracy_reward_action_threshold,
    "format": format_reward,
}


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user shows a robot observation image and asks for the next action. "
    "The Assistant first thinks about what the robot should do and then provides the action command. "
    "The reasoning process and action are enclosed within <think> </think> and <answer> </answer> tags, respectively. "
    "The action format is [x, y, z, gripper], where:\n"
    "  - x, y, z are position delta commands in range [-1, 1]\n"
    "  - gripper is -1 (open) or 1 (close)\n"
    "Example: <think>The cube is to the left, I should move left and grasp it</think>"
    "<answer>[0.5, 0.0, 0.0, 1.0]</answer>"
)


# ============================================================================
# Main Training Function
# ============================================================================

def main(script_args, training_args, model_args):
    """
    Main function for GRPO training on ManiSkill tasks
    """
    # Set reward functions for robot action prediction
    script_args.reward_funcs = ['accuracy_action_mse', 'format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    print(f"Using reward functions: {script_args.reward_funcs}")

    # Load dataset from JSON file
    print(f"Loading dataset from: {script_args.dataset_name}")
    dataset = load_dataset("json", data_files=script_args.dataset_name)

    # Format dataset into conversation format
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["prompt"]},
                    ],
                },
            ],
        }

    # Check if dataset has images (required for ManiSkill)
    if "image" in dataset[script_args.dataset_train_split].features:
        print("✓ Dataset contains images, using vision-language format")
        dataset = dataset.map(make_conversation_image)
    else:
        raise ValueError("Dataset must have 'image' field for ManiSkill robot tasks")

    # Select trainer class
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print(f"Using trainer: {trainer_cls.__name__}")

    # Initialize GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the trained model
    print(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    print("Training complete!")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Print debug info
    if os.getenv("DEBUG_MODE") == "true":
        print("=" * 60)
        print("DEBUG MODE ENABLED")
        print(f"Log path: {os.getenv('LOG_PATH')}")
        print("=" * 60)

    main(script_args, training_args, model_args)
