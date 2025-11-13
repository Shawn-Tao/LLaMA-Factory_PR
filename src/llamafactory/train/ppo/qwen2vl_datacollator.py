import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling

class DataCollatorForQwen2VL(DataCollatorForLanguageModeling):
    """
    针对 Qwen2.5-VL 的 DataCollator：
    - 处理已 flatten 的 pixel_values: (num_patches, patch_dim)
    - 对齐 image_grid_thw: (num_images, 3)
    - 支持 variable image count per sample
    """

    def __call__(self, examples):
        # 1️⃣ 文本部分 (标准)
        input_ids = [e["input_ids"] for e in examples]
        attention_mask = [e["attention_mask"] for e in examples]

        batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
        }

        # 2️⃣ 视觉特征 (Qwen2.5-VL flatten 格式)
        if "pixel_values" in examples[0]:
            # 计算所有样本的最大 patch 数
            max_patches = max(e["pixel_values"].shape[0] for e in examples)
            patch_dim = examples[0]["pixel_values"].shape[1]

            # pad pixel_values 到相同 patch 长度
            pixel_values_padded = []
            for e in examples:
                n_patches = e["pixel_values"].shape[0]
                pad_len = max_patches - n_patches
                if pad_len > 0:
                    pad = torch.zeros((pad_len, patch_dim), dtype=e["pixel_values"].dtype)
                    pixel_values_padded.append(torch.cat([e["pixel_values"], pad], dim=0))
                else:
                    pixel_values_padded.append(e["pixel_values"])
            batch["pixel_values"] = torch.stack(pixel_values_padded, dim=0)  # [B, max_patches, patch_dim]

            # 3️⃣ image_grid_thw: 对齐 num_images 维度
            max_imgs = max(e["image_grid_thw"].shape[0] for e in examples)
            grid_padded = []
            for e in examples:
                n_imgs = e["image_grid_thw"].shape[0]
                pad_len = max_imgs - n_imgs
                if pad_len > 0:
                    pad = torch.zeros((pad_len, 3), dtype=torch.long)
                    grid_padded.append(torch.cat([e["image_grid_thw"], pad], dim=0))
                else:
                    grid_padded.append(e["image_grid_thw"])
            batch["image_grid_thw"] = torch.stack(grid_padded, dim=0)

        # 4️⃣ 可选视觉辅助字段（如果存在）
        for key in ["position_ids", "rope_deltas"]:
            if key in examples[0]:
                batch[key] = pad_sequence([e[key] for e in examples], batch_first=True, padding_value=0)

        # PPO 不需要 labels
        batch.pop("labels", None)
        return batch
