# PPO调试记录

## 数据输入部分

### 数据文件格式：

最基础的函数是 loader.py文件中的 `_load_single_dataset()`, 其针对各种类型的数据来源进行了宽松的实现，我们前述针对`VLM-R1`制作的`jsonl`格式文件可以被直接读取。

### 数据读取方法的修改：

`_load_single_dataset()`函数会调用一个`convertor`将输入的`jsonl`处理为面向多模态输入需求的格式。

针对我们的VLN任务，我们新定义一个`convertor`，将`VLN`相关的数据进行组织。

```python

@dataclass
class VLNDatasetConverter(DatasetConverter):
    # ...... 省略
        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
            "goal_position": example["goal_position"] if "goal_position" in example else "",
            "distance_to_goal": example["distance_to_goal"] if "distance_to_goal" in example else "",
            "agent_position": example["agent_position"] if "agent_position" in example else "",
            "agent_heading": example["agent_heading"] if "agent_heading" in example else "",
        }
        return output

DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "vln": VLNDatasetConverter,
}

```

## 模型读取部分

trl 会增加一个 valuehead，llamafactory调用时，会在peftmodel外面再套一层壳，详见 `/LLM-VLM/LLaMA-Factory/src/llamafactory/model/loader.py` 文件中:

```python
if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")
```

这会导致数据处理部分，collator中，生成position_ids的代码失效，因为调用了 peftmodel内部的函数：


```python
        if self.get_rope_func is not None:
            if xxxxxxxx
            xxxxxx
            xxxxxx
            else:  # for qwen2vl
                features["position_ids"], features["rope_deltas"] = self.get_rope_func(**rope_index_kwargs)
        if (
            self.model is not None
            and getattr(self.model.config, "model_type", None)
            in ["glm4v", "qwen2_vl", "qwen2_5_vl", "qwen2_5_omni_thinker"]
            and ("position_ids" not in features or features["position_ids"].dim() != 3)
        ):
            raise ValueError("Qwen2-VL/Qwen2.5-Omni model requires 3D position ids for mrope.")
```

因此，做这样的处理:

```python
from trl import AutoModelForCausalLMWithValueHead

class Qwen2VLWithValueHead(AutoModelForCausalLMWithValueHead):
    def get_rope_index(self, *args, **kwargs):
        return self.pretrained_model.get_rope_index(*args, **kwargs)


if add_valuehead:
        # model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        model = Qwen2VLWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

```

这样，把方法从里面拿出来，拿到外面，这样就能过判定了


## 数据组织过程

数据组织过程由原始数据到网络完成推理，再到reward函数计算，直接按照 VLM-R1 要求的 jsonl 格式即可，在data——info中填写明白就可以。

数据首先由 `train/ppo/workflow.py` 中调用的 `get_dataset` 函数读入，

```python
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
```

该函数可以在 `LLaMA-Factory/src/llamafactory/data/loader.py` 文件中找到，会依次调用同一文件中的 `_load_single_dataset()` , `_get_merged_dataset` 等方法，完成原始数据文件的加载，
然后调用`_get_preprocessed_dataset` , 这一函数会调用 `_get_dataset_processor`获取到用于数据预处理的 `processor`,并调用 `processor` 对数据进行处理， 
`processor` 根据 配置文件中的 stage 字段来选取，我们这里选取的是自己定义的 `unsupervised.py` 文件中的 `PPO_VLNDatasetProcessor`, 这一类主要继承了无监督的方法，我们只是往里面加了对于特权信息的处理。

随后，回到 workflow.py 文件，数据进入了 `CustomPPOTrainer` 类对象 `ppo_trainer` 中，这之后，相关的实现看 `/LLaMA-Factory/src/llamafactory/train/ppo/trainer.py` 文件。
这一个类对象继承自 trl 库中的 PPOTrainer 类，训练过程llamafactory库沿用了很多 trl库中的实现，这里我们主要做了以下几个改进： 

1. 重载了了会导致特权信息丢失的方法 `prepare_dataloader`

```
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
```

2. 将gradient_accumulation_steps降低为1，因为 dataloader 抓取数据，经过qwen 对应的 collator处理后，会将图片信息等展平，具体可以参考 `collator.py` 与 `LLaMA-Factory/src/llamafactory/data/mm_plugin.py` 这两个文件。


## 奖励函数设置



