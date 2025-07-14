# -*- coding: utf-8 -*-
"""
PretrainedConfig类：
0. 该类包含了 模型结构、任务微调、tokenizer、Pytorch/tensorflow 等的各种配置信息
1.自定义模型通过继承该类来配置公共的transformers模型结构的元数据信息，如：num_hidden_layer、hidden_size、num_attention_heads
2.通过继承该类并新增属性如：num_key_value_heads，然后在模型中根据配置参数来实现 组注意力机制GQA。

"""

def test_pretrained_config():
    from transformers import PretrainedConfig
    from transformers import BertConfig, BertModel

    # 自定义一个配置：8层、512隐藏维度、8个注意力头
    config = BertConfig(
        num_hidden_layers=8,
        hidden_size=512,
        num_attention_heads=16
    )

    # 用这个 config 初始化模型
    model = BertModel(config)

    # 输出模型信息
    print(model)

"""
通过 Pretrained 封装 ResNet类。
"""

# 1. 自定义配置类
from transformers import PretrainedConfig
from typing import List

class ResnetConfig(PretrainedConfig):
    def __init__(self,
                 block_type="bottleneck",
                 layers: list[int] = [3, 4, 6, 3],
                 num_classes: int = 1000,
                 input_channels: int = 3,
                 cardinality: int = 1,
                 base_width: int = 64,
                 stem_width: int = 64,
                 stem_type: str = "",
                 avg_down: bool = False,
                 **kwargs,
                 ):
        if block_type not in ["bottleneck", "basic"]:
            raise ValueError(f"block_type must be one of 'bottleneck' or 'basic', got {block_type}")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"stem_type must be one of '', 'deep', got {stem_type}")
        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)  # 首先初始化指定的参数，然后将其他参数传入父类，完成初始化。


# 2. 根据自定义配置类来自定义模型结构(Resnet的骨架，用来提取特征)
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet

BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}

class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)


# 3. 通过 模型配置类和模型结构类 来创建模型。
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModel(resnet50d_config)


# 4.创建模型后并初始化参数后(timm默认提供初始化参数)，就可以训练/微调了。此处直接使用 与训练参数
import timm
pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())  # 初始化模型参数


# 5.将模型推送到 Huggingface Hub
## 1.设置网络代理 2.新建 模型配置类和模型类 文件 3.huggingface-cli login验证登录 5.写脚本调用push_to_hub()方法推送到HUB


