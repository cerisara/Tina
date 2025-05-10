from math_verify import LatexExtractionConfig, parse, verify
from transformers import TrainerCallback
import datasets
import torch
from datasets import Dataset, load_dataset
from datetime import datetime
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from tina.post_train_hf.grpo_trainer import GRPOTrainer
from latex2sympy2_extended import NormalizationConfig
import numpy as np
import xladder

modnom = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

REASON_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
# borrowed from https://github.com/knoveleng/open-rs/blob/main/recipes/grpo.yaml
SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} .\n The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Note that respond by English, NOT use other languages."

def make_conv_for_grpo(example, system_prompt):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
    }

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]

class GradientClippingLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        self.clipped_grad_norm = np.sqrt(sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["clipped_grad_norm"] = self.clipped_grad_norm

model_post_train_dataset_name = "knoveleng/open-s1"
train_dataset = load_dataset(model_post_train_dataset_name, split="train")
# required by GRPOTrainer: (prompt, solution) columns
if 'solution' not in train_dataset.column_names and 'answer' in train_dataset.column_names:
    train_dataset = train_dataset.rename_column('answer', 'solution')

    # Wrap the 'solution' values in $...$
    def wrap_in_math(example):
        return {"solution": f"${example['solution']}$"}

    # Apply the transformation to the entire dataset
    train_dataset = train_dataset.map(wrap_in_math)
if 'problem' not in train_dataset.column_names and 'question' in train_dataset.column_names:
    train_dataset = train_dataset.rename_column('question', 'problem')
if 'problem' not in train_dataset.column_names and 'prompt' in train_dataset.column_names:
    train_dataset = train_dataset.rename_column('prompt', 'problem')
if "messages" in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns("messages")

train_dataset = train_dataset.map(make_conv_for_grpo, fn_kwargs={"system_prompt": SYSTEM_PROMPT})


tokenizer = AutoTokenizer.from_pretrained(modnom)
tokenizer.pad_token = "<|fim_pad|>" # for Qwen
tokenizer.chat_template = REASON_CHAT_TEMPLATE

model = AutoModelForCausalLM.from_pretrained(modnom, torch_dtype=torch.bfloat16) #, attn_implementation="flash_attention_2", use_cache=False)
xladder.addLadder(model, 128, 4)

rl_reward_funcs = [accuracy_reward, format_reward]
reward_weights = (1.0, 2.0)

callbacks = [
            GradientClippingLoggerCallback(),
        ]

trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rl_reward_funcs,
        train_dataset=train_dataset,
        callbacks=callbacks)

train_result = trainer.train()


exit()

"""
# Etude de GRPOTrainer:

_get_train_sampler()
_get_eval_sampler() 
_get_per_token_logps() : get log-prob per token
_prepare_inputs() :
    tokenize prompt
    generate completion
    compute logits for completion with ref LLM
    compute reward on completion
    compute group rewards and then advantages
compute_loss : 
    call get log-prob per token
    calc KL(logprob(ref), logprob(model))
    loss = KL + advantages + log-prob
prediction_step :
    call _prepare_inputs & compute_loss

"""

