# ptp_utils.py

import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict
from PIL import Image

def save_images(images, dest, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    pil_img = Image.fromarray(images[-1])
    pil_img.save(dest)
    # display(pil_img)


def save_image(images, dest, num_rows=1, offset_ratio=0.02):
    print(images.shape)
    pil_img = Image.fromarray(images[0])
    pil_img.save(dest)


# UAC 的实现，对潜空间图像进行一次注意力计算
def register_attention_control(model, controller, use_lora):
    # 自注意力替换和交叉注意力替换的实现
    class AttnProcessor():
        def __init__(self, place_in_unet):
            self.place_in_unet = place_in_unet                                                                 # 指示注意力模块在 UNet 架构中的位置

        def __call__(self,                                                                                     # 定义了 __call__ 方法，使得 AttnProcessor 实例可以像函数一样被调用
            attn,                                # 注意力控制方法
            hidden_states,                       # 潜空间图像
            encoder_hidden_states=None,          # 文本嵌入
            attention_mask=None,
            temb=None,
            scale=1.0,
            iter_cur=0,
            phase="sample"):
            # The `Attention` class can call different attention processors / attention functions

            # 注意力计算的主体
            residual = hidden_states                                                                           # 保存原始的隐藏状态作为残差连接的一部分

            # 如果存在空间规范化，则应用之
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            # 调整隐藏状态的维度以适应后续操作
            input_ndim = hidden_states.ndim                                                                    # 获取隐藏状态张量的维度数
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape                                       # 解构 shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)        # (batch_size, channel, height, width) -->
                                                                                                               # (batch_size, height * width, channel) -->
                                                                                                               # (batch_size, sequence_length, channel)
            # print("hidden_states.shape = ", hidden_states.shape)

            # 准备注意力控制的参数
            h = attn.heads                                                                                     # 获取注意力头的数量
            is_cross = encoder_hidden_states is not None                                                       # 判断是否为交叉注意力 encoder_hidden_states 存在即为 交叉注意力

            # if encoder_hidden_states is not None:
            #     print("cross attention")
            if encoder_hidden_states is None:                                                                  # 是自注意力
                # print("self attention")
                encoder_hidden_states = hidden_states                                                          # encoder_hidden_states 设为自身
            elif attn.norm_cross:                                                                              # 是交叉注意力，说明有文本
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)                 # encoder_hidden_states 经过规范化处理


            batch_size, sequence_length, _ = (                                                                 # encoder_hidden_states 为空时使用 hidden_states 的形状来确定批量大小和序列长度
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape          # encoder_hidden_states 不为空时使用 encoder_hidden_states 来确定
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)          # 处理的注意力掩码张量填充至 sequence_length
            # 获得 q, k, v 分头:
            # q:    (batch_size, num_heads, sequence_length, head_dim)
            # k, v: (batch_size, num_heads, vocab_len, head_dim)
            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)

            # head_to_batch_dim: (batch_size, num_heads, sequence_length, head_dim)   -->    (batch_size * num_heads, sequence_length, head_dim)
            #                    (batch_size, num_heads, vocab_len, head_dim)         -->    (batch_size * num_heads, vocab_len, head_dim)
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)
            # 自注意力
            # 执行 SelfEdit 即 AttentionControlEdit 的 self_attn_forward()
            # 这一步替换了布局分支和目标分支在自注意力过程中的 q, k, v

            # 修改
            # q, k 的 shape 均为
            # (2 * batch_size * num_heads, sequence_length, head_dim)
            if not is_cross and phase == "sample":
                q, k, v = controller.self_attn_forward(q, k, v, attn.heads)                                                   # 获得源注意力 目标注意力 相互注意力的 q, k, v

            # 计算注意力得分 M_src, M_tgt, M_lay
            # 在计算的时候其实是计算 q, k 的相似度并用 softmax 转换为概率分布，attention_mask 会标记填充位置并在计算注意力分数的时候给他一个极小的权重，使其概率接近 0 
            attention_probs = attn.get_attention_scores(q, k, attention_mask)                                  # (batch_size * num_heads, sequence_length, vocab_len)

            # 交叉注意力
            # 执行 CrossEdit 即 AttentionControlEdit 里的 forward()
            if is_cross and phase == "sample":
                attention_probs  = controller(attention_probs, is_cross, self.place_in_unet)                   # 执行注意力替换操作

            hidden_states = torch.bmm(attention_probs, v)                                                      # 使用广播矩阵乘法（torch.bmm）将注意力得分应用于 v，得到加权后的隐藏状态
            # (batch_size * num_heads, sequence_length, vocab_len) * (batch_size * num_heads, vocab_len, head_dim) --> (batch_size * num_heads, sequence_length, head_dim)

            hidden_states = attn.batch_to_head_dim(hidden_states)                                              # 将 shape 转回去
            # batch_to_head__dim: (batch_size * num_heads, sequence_length, head_dim) --> (batch_size, num_heads, sequence_length, head_dim)

            # 第一个线性变换: 将多头合并
            # (batch_size, num_heads, sequence_length, head_dim) --> (batch_size, sequence_length, channel)
            hidden_states = attn.to_out[0](hidden_states)
            # 第二个线性变换: Dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                # (batch_size, sequence_length, channel) --> (batch_size, channel, sequence_length) --> (batch_size, channel, height, width)
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual                                                       # 将处理后的隐藏状态与 residual 相加

            hidden_states = hidden_states / attn.rescale_output_factor                                         # 输出重缩放

            return hidden_states                                                                               # 返回经过注意力控制的潜空间图像

    # 遍历神经网络子网络 net_ 并为所有的 Attention 类型的模块的注意力处理的 AttnProcessor 替换为上面自己定义的 AttnProcessor，返回统计的 Attention 层数
    def register_recr(net_, count, place_in_unet):
        for idx, m in enumerate(net_.modules()):                                     # 遍历 net_ 的所有子模块，包括 net_ 本身
            # print(m.__class__.__name__)
            if m.__class__.__name__ == "Attention":                                  # 对于每一个子模块 m，检查它的类名是否为 "Attention"
                count += 1                                                           # 如果是，就增加 count 计数器
                m.processor = AttnProcessor(place_in_unet)                           # 替换这个模块的 AttnProcessor，并传入 place_in_unet 参数
        return count                                                                 # 返回更新后的 count 值

    cross_att_count = 0                # 初始化累计注意力层的数量
    
    sub_nets = model.unet.named_children()

    if use_lora:
        unet = model.unet.model        # model.unet 的所有子网络的名称及其对应的模块对象 (module_name, nodule)
    else:
        unet = model.unet

    sub_nets = unet.named_children()
    # 遍历所有子网络，根据 down, up, mid 为每个子网络的子模块注册 AttnProcessor 并累加 Attention 模块的数量
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count                                      # 得到最后的累加注意力层的数量 在我这段代码上改
    print("controller.num_att_layers = ", controller.num_att_layers)


# 得到 word_place 在 text 中经过处理得到的所有分词索引
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")                                                                          # 输入的文本字符串 text 按照空格进行分割，得到一个包含各个单词的列表 split_text
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]                       # word 是要找的单词，找到它在 text 中的所有索引并放入列表中
    elif type(word_place) is int:
        word_place = [word_place]                                                                         # word 本身就是索引，放入列表中
    out = []                                                                                              # 存放每个单词的索引
    if len(word_place) > 0:                                                                               # 遍历所有 word
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]     # 使用 tokenizer 对整个文本 text 进行编码，然后解码每个编码单元，去除前缀 # 符号，得到一个去除首尾特殊标记后的单词列表 words_encode
        cur_len, ptr = 0, 0                          # 累计的字符长度；当前处理到的单词索引位置

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])          # 累加长度
            if ptr in word_place:                    # 如果当前处理的单词索引 ptr 在 word_place 列表中，则将当前单词在 words_encode 中的位置加 1 后添加到 out 列表中
                out.append(i + 1)                    # 记录 word_place 中对应位置经过 tokenizer 和解码后得到的分词索引
            if cur_len >= len(split_text[ptr]):      # 如果累计长度 cur_len 大于或等于当前单词 split_text[ptr] 的长度，则移动到下一个单词索引 ptr，并将累计长度 cur_len 清零
                ptr += 1                             # 已经处理完这个单词，移动到下个单词
                cur_len = 0
    return np.array(out)                             # 返回所有分词索引


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    return alpha_time_words
