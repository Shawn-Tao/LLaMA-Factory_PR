# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional
import time
from peft import PeftModel

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits,convert_to_scalar,WANDB_PADDING,stack_dicts,stats_to_np
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm

import torch
from typing import Callable, List, Optional, Union
from datasets import Dataset, load_dataset, load_from_disk

from .ppo_reward import RewardManager
from .qwen2vl_datacollator import DataCollatorForQwen2VL
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

# # 用来判断视觉分支有没有被执行的钩子函数
# def vision_hook(module, input, output):
#     print(">>> [视觉分支执行] Vision encoder output:", output[0].shape if isinstance(output, tuple) else output.shape)


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)

# ! 这个函数是罪魁祸首，调用了_remove_unused_columns函数，在最新版的trl中已经没有这个函数了，但是llama-factory依赖的是0.9.6版本的trl。
# ! 这里通过继承PPOTrainer并重写prepare_dataloader函数来避免调用_remove_unused_columns函数。
class VLPPOTrainer(PPOTrainer):
     def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        # if isinstance(dataset, Dataset):
        #     dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

# ! finetuning_args.reward_model_type is lora!!!!!!!
# class CustomPPOTrainer(PPOTrainer, Trainer):
class CustomPPOTrainer(VLPPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")
        
        # !dataloader的batch_size，来自于这里的 ppo_config.batch_size，finetuning_args.ppo_buffer_size 就是1
        # !这里backward_batch_size 是每个设备的 batch_size 乘以 gradient_accumulation_steps
        # !mini_batch_size 是每个设备的 batch_size
        
        # print(finetuning_args.ppo_buffer_size)
        # exit()

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        # backward_batch_size = training_args.per_device_train_batch_size
        
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)
        
        ppo_data_collator = DataCollatorForQwen2VL(tokenizer, mlm=False)
        
        # 这里生成了 self.dataloader.
        # ! trl 0.9.6版本中，会主动调用一个prepare_dataloader函数，会压缩数据集字段
        VLPPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
            training_data_collator=ppo_data_collator,
        )
        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=[self.processing_class.eos_token_id] + self.processing_class.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.processing_class, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)


    # def split_qwen2vl_batch(batch, num_images_per_sample):
    #     mini_batch = []
    #     grid_idx, patch_idx = 0, 0
    #     for b in range(len(num_images_per_sample)):
    #         n_img = num_images_per_sample[b]
    #         grids = batch["image_grid_thw"][grid_idx : grid_idx + n_img]
    #         patch_count = int(torch.prod(grids, dim=1).sum())
    #         sample = {
    #             "input_ids": batch["input_ids"][b:b+1],
    #             "attention_mask": batch["attention_mask"][b:b+1],
    #             "labels": batch["labels"][b:b+1],
    #             "position_ids": batch["position_ids"][:, b:b+1, :],
    #             "rope_deltas": batch["rope_deltas"][b:b+1],
    #             "pixel_values": batch["pixel_values"][patch_idx : patch_idx + patch_count],
    #             "image_grid_thw": grids,
    #         }
    #         priveleged_info_batch = {
    #             "goal_position": batch["goal_position"][b : b + 1],
    #             "distance_to_goal": batch["distance_to_goal"][b : b + 1],
    #             "agent_position": batch["agent_position"][b : b + 1],
    #             "agent_heading": batch["agent_heading"][b : b + 1],
    #         }
    #         mini_batch.append(sample)
    #         grid_idx += n_img
    #         patch_idx += patch_count
    #     return mini_batch,
    
    
    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")
        
        # ! dataloader调用了上文中的data_collator
        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)
        
        # ! 注册奖励函数类
        self.reward_manager = RewardManager()
        # ! 获取所有带后缀的奖励函数
        self.reward_funcs = [
            getattr(self.reward_manager, name) for name in dir(self.reward_manager) if callable(getattr(self.reward_manager, name)) and name.endswith("_reward")
        ]

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                # ! 这里会出点问题，需要改collator代码
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)
                
            # print("batch:", batch)
            # exit()

            # Get inputs
            self.model.eval()
            self.processing_class.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            
            # print(batch)
            # exit()
            # for k, v in batch.items():
            #     if isinstance(v, torch.Tensor):
            #         print(k, v.shape)
            #     else:
            #         print(k, type(v))
            # exit()
            
            # for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
            #     # mini_batch = {
            #     #     "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
            #     #     "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
            #     # }
            #     mini_batch = {
            #         "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
            #         "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
            #         "pixel_values": batch["pixel_values"][idx : idx + self.config.mini_batch_size],
            #         "image_grid_thw": batch["image_grid_thw"][idx : idx + self.config.mini_batch_size],
            #         "rope_deltas": batch["rope_deltas"][idx : idx + self.config.mini_batch_size],
            #         "position_ids": batch["position_ids"][idx : idx + self.config.mini_batch_size],
            #         "labels": batch["labels"][idx : idx + self.config.mini_batch_size],
            #     }
            #     priveleged_info_batch = {
            #         "goal_position": batch["goal_position"][idx : idx + self.config.mini_batch_size],
            #         "distance_to_goal": batch["distance_to_goal"][idx : idx + self.config.mini_batch_size],
            #         "agent_position": batch["agent_position"][idx : idx + self.config.mini_batch_size],
            #         "agent_heading": batch["agent_heading"][idx : idx + self.config.mini_batch_size],
            #     }
            #     # print(">>> decoded input_ids:", mini_batch["input_ids"][0])
            #     # print(">>> decoded input_ids:", self.processing_class.decode(mini_batch["input_ids"][0]))
            #     # print("vocab_size:", self.processing_class.vocab_size)
            #     # # exit()
            #     # print("labels:", mini_batch["labels"])
            #     # print(">>> decoded labels:", self.processing_class.decode(mini_batch["labels"][0]))
            #     # exit()
            #     mini_batch_queries, mini_batch_responses = self.get_inputs(mini_batch)
            #     print("mini_batch_queries:", mini_batch_queries[0])
            #     print("mini_batch_responses:", mini_batch_responses[0])
            #     print("labels:", mini_batch["labels"])
            #     for i, q in enumerate(mini_batch_queries):
            #         print(f">>> decoded query[{i}]:", self.processing_class.decode(q, skip_special_tokens=True))
            #     for i, r in enumerate(mini_batch_responses):
            #         print(f">>> decoded response[{i}]:", self.processing_class.decode(r, skip_special_tokens=True))
            #     print(">>> decoded label:", self.processing_class.decode(mini_batch["labels"][0]))
            #     exit()
            #     # continue
            #     mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
            #     queries.extend(mini_batch_queries)
            #     responses.extend(mini_batch_responses)
            #     rewards.extend(mini_batch_rewards)
            # # exit()
            
            mini_batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "pixel_values": batch["pixel_values"],
                "image_grid_thw": batch["image_grid_thw"],
                "rope_deltas": batch["rope_deltas"],
                "position_ids": batch["position_ids"],
                "labels": batch["labels"],
            }
            priveleged_info_batch = {
                "goal_position": batch["goal_position"],
                "distance_to_goal": batch["distance_to_goal"],
                "agent_position": batch["agent_position"],
                "agent_heading": batch["agent_heading"],
            }
            
            # test_batch = mini_batch.pop("labels")
            
            # with self.amp_context:  # support bf16
            #     logits, _, values = self.model(**test_batch, return_dict=True, use_cache=False)
                
            # print("logits shape:", logits.shape)
            
            # # ! 看一下这里有没有pad -- 看过了，似乎是没有的
            
            # ! 这里的处理, 把input_ids单独拿出来了,还减少了一个维度,相当于做了 squeeze
            mini_batch_queries, mini_batch_responses = self.get_inputs(mini_batch)
            
            
            # queries.extend(mini_batch_queries)
            # responses.extend(mini_batch_responses)
            # input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            
            # print("input_ids:", input_ids)
            # print("decoded input_ids:", [self.processing_class.decode(ids, skip_special_tokens=True) for ids in input_ids])

            labels_cpu = mini_batch["labels"].detach().to("cpu")
            
            for k, v in priveleged_info_batch.items():
                    priveleged_info_batch[k] = v.detach().cpu()
            mini_batch_rewards = self.get_ppo_rewards(mini_batch_queries, mini_batch_responses, labels_cpu, priveleged_info_batch)
            
            mini_batch_rewards = [mini_batch_rewards]
            
            
            # ! 原本只extend了文本信息，我们这里需要把视觉信息等全部包含进去
            # 直接extend会出问题
            # queries.extend(mini_batch_queries)
            # responses.extend(mini_batch_responses)
            # rewards.extend(mini_batch_rewards)
            
            queries = [mini_batch]
            responses = mini_batch_responses
            rewards = mini_batch_rewards
            
            # 将模型切换到训练模式
            self.model.train()
            #! Run PPO step !self.step 是trl父类中的函数，应该会调用batched_forward_pass
            stats = self.step(queries, responses, rewards)
            
            self.processing_class.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.processing_class.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.processing_class.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    #! 主要是利用 query 通过模型推理，生成response
    @torch.no_grad()
    def get_inputs(self, batch: dict[str, "torch.Tensor"]) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries."""
        # 裁剪输入的padding部分--仅针对 input_ids 和 attention_mask
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.processing_class.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                if(k in ["input_ids", "attention_mask"]):
                    batch[k] = v[:, start_index:]
        
        # type(self.model) --->  <class 'torch.nn.parallel.distributed.DistributedDataParallel'>
        
        # 进入原生模型
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            # unwrapped_model -> <class 'llamafactory.model.loader.Qwen2VLWithValueHead'>
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            
            # base_model = unwrapped_model.pretrained_model if hasattr(unwrapped_model, "pretrained_model") else unwrapped_model.model

            # print(type(unwrapped_model))
            
            # def register_vision_hook(model):
            #     """自动查找并挂钩视觉分支"""
            #     def vision_hook(module, input, output):
            #         print(">>> [视觉分支执行] Vision encoder output:", 
            #             output[0].shape if isinstance(output, tuple) else output.shape)

            #     vision_module = None
            #     for name, submodule in model.named_modules():
            #         if "vision" in name or "visual" in name:
            #             vision_module = submodule
            #             break

            #     if vision_module is not None:
            #         print(f"✅ 找到视觉模块: {name}")
            #         handle = vision_module.register_forward_hook(vision_hook)
            #         return handle
            #     else:
            #         print("⚠️ 未找到视觉模块，请检查模型结构（可能是纯文本模型）")
            #         return None
            
            # handle = register_vision_hook(unwrapped_model)
            
            # 暂时 upcast LayerNorm 部分fp16->fp32 防止梯度爆炸
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            # !执行生成过程-->模型推理，生成结果
            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            
            # 恢复 LayerNorm 参数
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)
                
            # # === (7) 注销 Hook（防止内存泄漏） ===
            # if handle is not None:
            #     handle.remove()
            #     print(">>> 🧹 已移除视觉分支 hook")

        # 切分 query / response，两者都搬到 CPU，detach 以避免梯度回传。
        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        
        # ! 以下的处理还减小了维度,相当于做了一次 squeeze
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.processing_class.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.processing_class.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.processing_class.eos_token_id == self.processing_class.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses

    # ! 尝试修改这里，把我们在VLM-R1中定义的奖励函数都搬运过来，计算最终的score
    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.processing_class.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")
            
        # if self.finetuning_args.reward_model_type == "vln":
        #     pass
            

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type
    
    # 自定义reward计算函数
    @torch.no_grad()
    def get_ppo_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        labels_cpu, 
        priveleged_info_batch,
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        
        reward_results = []
        for func in self.reward_funcs:
            reward_result = func(self.processing_class.decode(responses[0][:-1]),self.processing_class.decode(labels_cpu.squeeze()[:-2]) ,priveleged_info_batch)  # 调用函数
            reward_results.append(reward_result)
            
        # print("Found reward functions:", [f.__name__ for f in self.reward_funcs])
        # print("**************************Reward results:", reward_results)
        
        # reward_results to 1d tensor
        reward_tensor = torch.tensor(reward_results, dtype=torch.float32)
        reward_tensor = torch.sum(reward_tensor).unsqueeze(0)  # sum and make it 1d tensor
        
        return reward_tensor

    # ! 核心函数

    # @override
    # @PPODecorators.empty_device_cache()
    # def batched_forward_pass(
    #     self,
    #     model: "AutoModelForCausalLMWithValueHead",
    #     queries: "torch.Tensor",
    #     responses: "torch.Tensor",
    #     model_inputs: dict[str, Any],
    #     return_logits: bool = False,
    #     response_masks: Optional["torch.Tensor"] = None,
    # ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
    #     r"""Calculate model outputs in multiple batches.

    #     Subclass and override to inject custom behavior.
    #     """
    #     bs = len(queries)
    #     fbs = self.config.mini_batch_size
    #     all_logprobs = []
    #     all_logits = []
    #     all_masks = []
    #     all_values = []

    #     for i in range(math.ceil(bs / fbs)):
    #         input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
    #         query_batch = queries[i * fbs : (i + 1) * fbs]
    #         response_batch = responses[i * fbs : (i + 1) * fbs]
    #         if response_masks is not None:
    #             response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
    #         input_ids = input_kwargs["input_ids"]
    #         attention_mask = input_kwargs["attention_mask"]

    #         # ! value是critic——head输出的每个token的value估计。
    #         # ! 优势函数计算在trl中的父类 ppotrainer.step中实现
    #         with self.amp_context:  # support bf16
    #             logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

    #         logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    #         masks = torch.zeros_like(attention_mask)
    #         masks[:, :-1] = attention_mask[:, 1:]

    #         for j in range(len(query_batch)):
    #             start = len(query_batch[j]) - 1
    #             if attention_mask[j, 0] == 0:  # offset left padding
    #                 start += attention_mask[j, :].nonzero()[0].item()
    #             end = start + len(response_batch[j])

    #             if response_masks is not None:
    #                 response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

    #             masks[j, :start] = 0
    #             masks[j, end:] = 0
    #             if response_masks is not None:
    #                 masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

    #         if return_logits:
    #             all_logits.append(logits)
    #         else:
    #             del logits

    #         all_values.append(values)
    #         all_logprobs.append(logprobs)
    #         all_masks.append(masks)

    #     return (
    #         torch.cat(all_logprobs),
    #         torch.cat(all_logits)[:, :-1] if return_logits else None,
    #         torch.cat(all_values)[:, :-1],
    #         torch.cat(all_masks)[:, :-1],
    #     )
    
    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        # for i in range(math.ceil(bs / fbs)):
        #     input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
        #     query_batch = queries[i * fbs : (i + 1) * fbs]
        #     response_batch = responses[i * fbs : (i + 1) * fbs]
        #     if response_masks is not None:
        #         response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
        
        input_kwargs = model_inputs
        query_batch = queries
        response_batch = responses
        
        # print(">>> shapes before forward:")
        # print(" input_ids:", input_kwargs["input_ids"].shape)
        # print(" attention_mask:", input_kwargs["attention_mask"].shape)
        # if "position_ids" in input_kwargs:
        #     print(" position_ids:", input_kwargs["position_ids"].shape)
        # if "rope_deltas" in input_kwargs:
        #     print(" rope_deltas:", input_kwargs["rope_deltas"].shape)
        # if "pixel_values" in input_kwargs:
        #     print(" pixel_values:", input_kwargs["pixel_values"].shape)
        # if "image_grid_thw" in input_kwargs:
        #     print(" image_grid_thw:", input_kwargs["image_grid_thw"].shape, "sum(prod):", int(torch.prod(input_kwargs["image_grid_thw"], dim=1).sum()))
        # exit()
        
        input_ids = input_kwargs["input_ids"]
        attention_mask = input_kwargs["attention_mask"]
        
        with self.amp_context:  # support bf16
            logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)
            
        # exit()

        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        masks = torch.zeros_like(attention_mask)
        masks[:, :-1] = attention_mask[:, 1:]

        for j in range(len(query_batch)):
            start = len(query_batch[j]) - 1
            if attention_mask[j, 0] == 0:  # offset left padding
                start += attention_mask[j, :].nonzero()[0].item()
            end = start + len(response_batch[j])

            if response_masks is not None:
                response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

            masks[j, :start] = 0
            masks[j, end:] = 0
            if response_masks is not None:
                masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

        if return_logits:
            all_logits.append(logits)
        else:
            del logits

        all_values.append(values)
        all_logprobs.append(logprobs)
        all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
     
    @override        
    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`response_length`)
        Returns:
            `tuple`: The input processed data.
        """

        
        for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        
        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]
        masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores, masks
    
    def prepare_qwen_model_inputs(self, queries, responses):
        
        print(queries[0]["input_ids"].size())
        print(responses[0].size())
        
        base_model = getattr(self.model, "module", self.model)
        
        if hasattr(base_model, "get_rope_index"):
            self.get_rope_func = base_model.get_rope_index
            print("Using base_model.get_rope_index")
        elif hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            self.get_rope_func = base_model.model.get_rope_index
            print("Using base_model.model.get_rope_index")
        else:
            print("No get_rope_index found")
            self.get_rope_func = None
        
        #! 拼接 query 的 input_ids 和 response
        examples = []
        for q, r in zip(queries, responses):
            input_ids = q["input_ids"][0]
            q["input_ids"] = torch.cat([input_ids, r]).unsqueeze(0)
            q["attention_mask"] = torch.ones_like(q["input_ids"])

            if self.get_rope_func is not None:
                rope_index_kwargs = {
                    "input_ids": q["input_ids"],
                    "image_grid_thw": q.get("image_grid_thw"),
                    "attention_mask": (q["attention_mask"] >= 1).float(),
                }
                if "second_per_grid_ts" in q:  # for qwen2vl
                    rope_index_kwargs["second_per_grid_ts"] = q.get("second_per_grid_ts")

                q["position_ids"], q["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)

            if (
                base_model is not None
                and getattr(base_model.config, "model_type", None)
                in ["glm4v", "qwen2_vl", "qwen2_5_vl", "qwen2_5_omni_thinker"]
                and ("position_ids" not in q or q["position_ids"].dim() != 3)
            ):
                raise ValueError("Qwen2-VL/Qwen2.5-Omni model requires 3D position ids for mrope.")
            
            print(q["position_ids"])
            
            print(">>> shapes before forward:")
            print(" input_ids:", queries[0]["input_ids"].shape)
            print(" attention_mask:", queries[0]["attention_mask"].shape)
            if "position_ids" in queries[0]:
                print(" position_ids:", queries[0]["position_ids"].shape)
            if "rope_deltas" in queries[0]:
                print(" rope_deltas:", queries[0]["rope_deltas"].shape)
            if "pixel_values" in queries[0]:
                print(" pixel_values:", queries[0]["pixel_values"].shape)
            # if "image_grid_thw" in queries[0]:
            #     print(" image_grid_thw:", queries[0]["image_grid_thw"].shape, "sum(prod):", int(torch.prod(q["q"], dim=1).sum()))
        
            # exit()

            
            
            # # 更新 position_ids
            # if "position_ids" in q:
            #     old_pos = q["position_ids"]  # shape: [1, 1, seq_len_q]
            #     # old_pos 保存到本地 txt文档
            #     import os
            #     if not os.path.exists("debug"):
            #         os.makedirs("debug")
            #     with open("debug/old_position_ids.txt", "w") as f:
            #         f.write(str(old_pos.tolist()))
            #     exit()
                
                
            #     seq_len_q = old_pos.shape[-1]
            #     seq_len_r = r.shape[-1]
            #     new_positions = torch.arange(seq_len_q, seq_len_q + seq_len_r, device=old_pos.device).unsqueeze(0).unsqueeze(0)
            #     print("old_pos:",old_pos)
            #     print("new_positions:",new_positions)
            #     q["position_ids"] = torch.cat([old_pos, new_positions], dim=-1)
            

            q.pop("labels", None)
            examples.append(q)
        
            
        return examples
            
        # print("input_ids:", queries[0]["input_ids"])
        # input_ids = queries[0]["input_ids"][0].to("cpu")
        # print("decoded input_ids:", self.processing_class.decode(input_ids, skip_special_tokens=False))
        
        # examples = []
        # for idx, ids in enumerate(input_ids):
        #     examples.append({
        #         "input_ids": ids,
        #         "attention_mask": torch.ones_like(ids),
        #         "pixel_values": self.pixel_values[idx],       # (num_patches, patch_dim)
        #         "image_grid_thw": self.image_grid_thw[idx],   # (num_images, 3)
        #         "position_ids": self.position_ids[idx],
        #         "rope_deltas": self.rope_deltas[idx],
        #     })
        # batch = self.data_collator(examples).to(self.current_device)
        # batch.pop("labels", None)
        # return batch
        

           
    @override 
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size
        
        # queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]
        
        # queries, responses, scores, response_masks = self._step_safety_checker(
        #     bs, queries, responses, scores, response_masks
        # )
        

        
        scores = torch.tensor(scores, device=self.current_device)
        # # ! 测试出来目前use_score_scaling 是 false
        # if self.config.use_score_scaling:
        #     # Score scaling
        #     scores_mean, scores_std = self.running.update(scores)
        #     tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
        #     score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
        #     if self.config.use_score_norm:
        #         scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
        #     else:
        #         scores /= score_scaling_factor

        # # score_clip
        # # ! 测试出来目前没有
        # if self.config.score_clip is not None:
        #     # Score clipping
        #     scores_dtype = scores.dtype
        #     scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # # ! 测试出来目前没有启用
        # # if we want to push best model to the hub
        # if hasattr(self, "highest_reward"):
        #     if self.compare_step % self.config.compare_steps == 0:
        #         curr_mean_reward = scores.mean()
        #         # if the best reward ever seen
        #         if curr_mean_reward > self.highest_reward:
        #             self.highest_reward = curr_mean_reward
        #             # push model to hub
        #             self.push_to_hub(**self.push_to_hub_kwargs)
        #     self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        # model_inputs = self.prepare_model_inputs(queries, responses)
        model_inputs =  self.prepare_qwen_model_inputs(queries, responses)
        model_inputs = model_inputs[0]  # 目前只支持 per_device_batch_size = 1
        # print(model_inputs)

        # 是进行分布式训练的
        if self.is_distributed:
            pad_first = self.processing_class.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"], dim=1, pad_index=self.processing_class.pad_token_id, pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            
            #     # 如果是多模态模型，还要对 pixel_values 进行 pad
            # if "pixel_values" in model_inputs:
            #     model_inputs["pixel_values"] = self.accelerator.pad_across_processes(
            #         model_inputs["pixel_values"],
            #         dim=0,
            #         pad_index=0.0,  # 图像patch一般补零
            #         pad_first=False,
            #     )

            # # 对 image_grid_thw 同步 pad
            # if "image_grid_thw" in model_inputs:
            #     model_inputs["image_grid_thw"] = self.accelerator.pad_across_processes(
            #         model_inputs["image_grid_thw"],
            #         dim=0,
            #         pad_index=0,
            #         pad_first=False,
            #     )
                
            # if self.get_rope_func is not None:
                
            #     rope_index_kwargs = {
            #         "input_ids": model_inputs["input_ids"],
            #         "image_grid_thw": model_inputs.get("image_grid_thw"),
            #         "attention_mask": (model_inputs["attention_mask"] >= 1).float(),
            #     }
            #     if "second_per_grid_ts" in model_inputs:  # for qwen2vl
            #         rope_index_kwargs["second_per_grid_ts"] = model_inputs.get("second_per_grid_ts")

            #     model_inputs["position_ids"], model_inputs["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)
            
            # if self.is_encoder_decoder:
            #     model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
            #         model_inputs["decoder_input_ids"],
            #         dim=1,
            #         pad_index=self.processing_class.pad_token_id,
            #         pad_first=pad_first,
            #     )
            #     model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
            #         model_inputs["decoder_attention_mask"],
            #         dim=1,
            #         pad_index=0,
            #         pad_first=pad_first,
            #     )

        model_inputs_names = list(model_inputs.keys())
        
        # print(model_inputs)
        # exit()

        full_kl_penalty = self.config.kl_penalty == "full"
        
        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats