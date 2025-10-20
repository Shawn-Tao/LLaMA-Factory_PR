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