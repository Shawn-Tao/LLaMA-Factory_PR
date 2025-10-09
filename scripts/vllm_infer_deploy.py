# Copyright 2025 the LlamaFactory team.
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

import gc
import json

import argparse

from typing import Optional
from tqdm import tqdm

import fire
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from llamafactory.data.processor import UnsupervisedDatasetProcessor

import socket
import numpy as np
import struct
import pickle
import time

from utils.tcp_recver import CommandImageReceiver
# import cv2
# import imageio
from PIL import Image
import os


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

# template_header_str = f"Imagine you are a robot programmed for navigation tasks. You have given "
# temlpate_history_str = f"a video of historical obervations: <history_image> and "
# template_current_str = f"current observation: <image>. "
# template_task_str = f"Your assigned task is: \"<task>\". Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if task is completed."


template_header_str = f"You are a robot programmed for navigation tasks. You have given "
temlpate_history_str = f"serial of historical obervations: <history_image> and "
template_current_str = f"current observation: <image>. "
template_last_action_str = f"The sequential action decisions you made earlier are: \"<action>\"."
template_task_str = f"Your assigned task is: \"<task>\". Analyze the above information and decide your next action: whether to move forward a specific distance, turn left or right by a specific angle, or determine that the task is complete and stop."

history_action_list = []

def generage_prompt(image_list, instruction_str):
    
    header_str = ""
    history_str = ""
    current_str = ""
    task_str = ""
    
    if(len(image_list)==1):
        header_str = template_header_str
        current_str = template_current_str
        task_str = f'{template_task_str}'.replace("<task>", instruction_str)
    else:
        historical_image_str = ""
        for i in range(len(image_list)-1):
            historical_image_str += f" <image>"
        header_str = template_header_str
        history_str = temlpate_history_str.replace("<history_image>",historical_image_str)
        current_str = template_current_str
        task_str = template_task_str.replace("<task>", instruction_str)
    
    prompt = f"{header_str}{history_str}{current_str}{task_str}"
    
    return prompt

def generage_prompt_with_aciton(image_list, instruction_str):
    
    header_str = ""
    history_str = ""
    current_str = ""
    task_str = ""
    last_action_str = ""
    
    if(len(image_list)==1):
        header_str = template_header_str
        current_str = template_current_str
        task_str = f'{template_task_str}'.replace("<task>", instruction_str)
        history_action_list.clear()
    else:
        historical_image_str = ""
        for i in range(len(image_list)-1):
            historical_image_str += f" <image>"
        header_str = template_header_str
        history_str = temlpate_history_str.replace("<history_image>",historical_image_str)
        current_str = template_current_str
        task_str = template_task_str.replace("<task>", instruction_str)
    if len(history_action_list) > 0:
        last_action_str = template_last_action_str.replace("<action>", ",".join(history_action_list))
        # history_action_list.append(instruction_str)
        
    
    prompt = f"{header_str}{history_str}{current_str}{last_action_str}{task_str}"
    
    return prompt

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len

def _preprocess_image(self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

def _regularize_images(self, images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            if isinstance(image, (str, BinaryIO)):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

def template_and_tokenrize(prompt, 
                           images,
                           data_args,
                           template: "Template",
                           tokenizer: "PreTrainedTokenizer",
                           processor: Optional["ProcessorMixin"] = None 
                          ):
    
    # {'_prompt': [{'content': 'Imagine you are a robot programmed for navigation tasks. You have given current observation: <image>. Your assigned task is: "Walk towards the white bookshelf. Turn left at the white bookshelf. Stop in between the two tables. ". Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree, moving forward a certain distance, or stop if task is completed.', 'role': 'user'}], '_response': [{'content': 'turn right 45 degrees', 'role': 'assistant'}], '_system': '', '_tools': '', '_images': ['r2r_train_320/9224/9224_0.jpg'], '_videos': None, '_audios': None}
    
    messages = [{'content':prompt, 'role': 'user'},{'content': '', 'role': 'assistant'}]
    # images = ["test"]
    videos = []
    audios = []
    
    system = ""
    tools = ""      
    
    # print(messages)
    messages = template.mm_plugin.process_messages(messages, images, videos, audios, processor)
    print(messages)
    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    # print(input_ids)
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    input_ids, _ = template.mm_plugin.process_token_ids(
        input_ids, None, images, videos, audios, tokenizer, processor
    )
    # print(input_ids)
    source_len, target_len = infer_seqlen(len(input_ids), len(labels), data_args.cutoff_len)
    input_ids = input_ids[:source_len]
    # labels = labels[:target_len]
    return input_ids, labels

# 线性等距采样-- 当n大于2时，会取首尾，中间线性等距采样，
def uniform_sample(arr, n):
    indices = np.linspace(0, len(arr)-1, n, dtype=int)  # 生成均匀分布的索引
    return [arr[i] for i in indices]

def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 4096,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 16,
    max_image_buffer_size = 10,
    # condition = None,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )
    
    # exit()

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "dtype":"float16",        # open it, if GPUs do not support the bfloat16
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 10, "video": 0, "audio": 0}
    
    
    engine_args["mm_processor_kwargs"]={
            "min_pixels": 2 * 2 * 28 * 28,
            # "max_pixels": 320 * 320 * 28 * 28
            # "max_pixels": 320 * 320
            "max_pixels": 424 * 240
        }

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
        
    # print(engine_args)
    
    # exit()

    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None
        
    # 开启TCP服务器
    receiver = CommandImageReceiver()
        
    # udp_condition.acquire()
    current_instruction = ""
    
    infer_data_save_dir = "/LLM-VLM/infer_data_bak" 
    
    image_buffer_list = []
    max_image_buffer_size = 10
    
    save_time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    save_dir = f"{infer_data_save_dir}/{save_time_stamp}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    infer_count = 0
    image_count = 0
    save_data_flag = True
    infer_round_dir = ''
    
    # 注意以下分辨率处理
    while True:
        vllm_inputs, prompts, labels = [], [], []
        
        images, command = receiver.get_next_command_image()
        print("recv cmd and images: ", command, len(images))
        if command != None:
            current_instruction = command
            image_buffer_list.clear()
            
            if (save_data_flag == True):
                infer_count += 1
                infer_round_dir = f"{save_dir}/infer_round_{infer_count}"
                if not os.path.exists(infer_round_dir):
                    os.makedirs(infer_round_dir)
                image_count = 0
                
        for i in range(len(images)):
            image_buffer_list.append(images[i])
            
            if (save_data_flag == True):
                image_path = f"{infer_round_dir}/image_{image_count}.jpg"
                pil_image = Image.fromarray(images[i])
                pil_image.save(image_path)
                # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(image_path, image_bgr)
                image_count +=1
            
        # while(len(image_buffer_list) > max_image_buffer_size):
        #     image_buffer_list.pop(0)
        
        input_image_buffer = []
        if len(image_buffer_list) <= max_image_buffer_size:
            input_image_buffer = image_buffer_list
        else:
            # 线性等距采样
            input_image_buffer = uniform_sample(image_buffer_list, max_image_buffer_size)
        
        # 参考 https://docs.vllm.ai/en/latest/api/vllm/multimodal/inputs.html#vllm.multimodal.inputs.HfImageItem
        # 输入格式支持 PIL.Image, ndarray, torch.Tensor.
        multi_modal_data = {"image": input_image_buffer}
        
        prompt = generage_prompt_with_aciton(input_image_buffer, current_instruction)
        
        # 在这里引入 tokenizer， 参考 loader 里的 _get_preprocessed_dataset，， 首先用 prompt构建 dataset类，然后扔进去应该就行。参考之前的dataset构建，构建一个一样的 应该就行，
        # 打印一下之前的dataset中某一条，保证整体结构的情况下，在这里生成一条一样的应该就行
        input_ids, labels = template_and_tokenrize(prompt, input_image_buffer, data_args, template_obj ,**tokenizer_module)
        
        # print(tokenizer.decode(input_ids))
        # exit()
        
        vllm_inputs.append({"prompt_token_ids": input_ids, "multi_modal_data": multi_modal_data})
        
        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

        preds = [result.outputs[0].text for result in results]
        
        receiver._send_response(True, preds[0])
        
        history_action_list.append(preds[0])
        if len(history_action_list) > 1:
            history_action_list.pop(0)
        
        # with open(save_name, "a", encoding="utf-8") as f:
        #     for text, pred, label in zip(prompts, preds, labels):
        #         f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
        
        # for i in range(len(image_buffer_list)):
        #     # 判断图像分辨率，如果分辨率不是 112x56， 则进行缩放
        #     if image_buffer_list[i].shape[1] != 112 or image_buffer_list[i].shape[0] != 56:
        #         pil_img = Image.fromarray(image_buffer_list[i])
        #         pil_img_resized = pil_img.resize((112, 56), resample=Image.LANCZOS)
        #         # pil_img_resized = pil_img.resize((112, 56), Image.Resampling.BICUBIC)
        #         scaled_image = np.array(pil_img_resized)
        #         image_buffer_list[i] = scaled_image
        
        
        
if __name__ == "__main__":
    
    # deal with the argument
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="path to model",
    )
    
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="template",
    )
    
    parser.add_argument(
        "--max_image_buffer_size",
        type=int,
        required=True,
        help="max_image_buffer_size",
    )
    
    args = parser.parse_args()
    
    vllm_infer(model_name_or_path = args.model_name_or_path, template=args.template, max_image_buffer_size=args.max_image_buffer_size)
    
    
    # vllm_infer(model_name_or_path="/LLM-VLM/train_saves/R2R-240-5-20epoch-20250612/qwen2_vl-3b/outputs", template = "qwen2_vl", dataset ="r2r_train_240_5")

