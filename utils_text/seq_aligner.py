# seq_aligner.py

import torch
import copy
import torch.nn.functional as F
import numpy as np


# 存储与序列比对评分相关的参数，如插入/删除（gap）的得分、匹配字符（match）的得分以及不匹配字符（mismatch）的得分。
class ScoreParams:

    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match
        

# 没有被使用
def get_matrix(size_x, size_y, gap):          # 假设 size_x, size_y 应该是数组
    matrix = []
    # 创建一个大矩阵，里面每个元素是只有一个元素的小矩阵 [0]
    for i in range(len(size_x) + 1):
        sub_matrix = []
        for j in range(len(size_y) + 1):
            sub_matrix.append(0)
        matrix.append(sub_matrix)
    for j in range(1, len(size_y) + 1):
        matrix[0][j] = j * gap
    for i in range(1, len(size_x) + 1):
        matrix[i][0] = i * gap
    return matrix


# 创建一个评分矩阵 （gap 为插入或删除一个字符的代价）
def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)     # 创建一个 (size_x + 1, size_y + 1) 的全 0 矩阵
    # 初始评分矩阵
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix
    #    假设 gap 为 -2
    #    [[  0, -2, -4, -6, -8 ]
    #     [ -2,  0,  0,  0,  0 ]
    #     [ -4,  0,  0,  0,  0 ]
    #     [ -6,  0,  0,  0,  0 ]
    #     [ -8,  0,  0,  0,  0 ]]


# 创建一个回溯矩阵
def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix
    #
    #    [[ 4,  1,  1,  1,  1 ]
    #     [ 2,  0,  0,  0,  0 ]
    #     [ 2,  0,  0,  0,  0 ]
    #     [ 2,  0,  0,  0,  0 ]
    #     [ 2,  0,  0,  0,  0 ]]

# 全局对齐算法 （score 是一个包含 match, mismatch, gap 的评分规则）
def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)                                   # 获得初始评分矩阵
    trace_back = get_traceback_matrix(len(x), len(y))                                # 获得初始回溯矩阵
    # 动态规划填充矩阵
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            # 比较从左边、上方以及对角线方向过来的最大得分
            left = matrix[i, j - 1] + score.gap                                      # 这表示从左侧过来，在 y 序列中插入一个 gap
            up = matrix[i - 1, j] + score.gap                                        # 这表示从上侧过来，在 x 序列中插入一个 gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])   # 这表示两个字符之间的比较，判断它们是匹配还是不匹配。 匹配 + match 得分，不匹配 + mismatch 得分  x[i - 1], y[j - 1] 为对应位置的字母
            matrix[i, j] = max(left, up, diag)                                       # 找到评分最高的方法
            if matrix[i, j] == left:                                                 # 从左侧来为 1
                trace_back[i, j] = 1
            elif matrix[i, j] == up:                                                 # 从上侧来为 2
                trace_back[i, j] = 2
            else:                                                                    # 从对角线来为 3
                trace_back[i, j] = 3
    return matrix, trace_back


#  返回对齐后的 x, y, mapper_y_to_x
def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)                                 # 从最后开始回溯
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:              # 对角线
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:            # 左侧
            x_seq.append('-')
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:            # 上侧
            x_seq.append(x[i - 1])
            y_seq.append('-')
            i = i - 1
        elif trace_back[i][j] == 4:            # 4 则退出
            break
    mapper_y_to_x.reverse()                    # 反转
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


# 返回 mapper, alphas, m, alpha_e, alpha_m
# mapper 的构建过程包括了全局对齐算法的执行，它记录了 y_seq 中的每个元素在 x_seq 中的最佳匹配位置
# alphas 记录了 y_seq 中的元素是否在 x_seq 中找到了匹配的位置。通过全局对齐算法，我们可以确定哪些位置是匹配的，然后将这些位置标记为 1
# m 是 alphas 的深拷贝，通常与 alphas 保持一致
# alpha_e 记录了 目标混合词 local_prompt 中的 token 与 y_seq 中的元素是否匹配，对应匹配的位置 i 置 1，y[i] 则是目标混合词
# alpha_m 记录了 源混合词  mutual_prompt 中的 token 与 y_seq 中的元素是否匹配，对应匹配的位置 i 置 1，y[i] 则是源混合词
def get_mapper(x: str, y: str, specifier, tokenizer, encoder, device, max_len=77):
    # 对源提示，目标提示，目标混合词，源混合词编码
    locol_prompt, mutual_prompt = specifier                                           # 从 specifier 元组中提取出目标混合词和源混合词
    x_seq = tokenizer.encode(x)                                                       # 对 x 编码
    y_seq = tokenizer.encode(y)                                                       # 对 y 编码
    e_seq = tokenizer.encode(locol_prompt)                                            # 对 local_prompt 编码
    m_seq = tokenizer.encode(mutual_prompt)                                           # 对 mutual_prompt 编码

    # 设置罚分并对源提示，目标提示计算全局对齐
    score = ScoreParams(0, 1, -1)                                          # gap = 0, match = 1, mismatch = -1
    matrix, trace_back = global_align(x_seq, y_seq, score)                            # 得到 x 和 y 全局对齐的分数矩阵和回溯矩阵
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]                 # 得到对齐后 y 对 x 的位置映射表
                                                                                      # mapper_base = [(0, 0), (1, -1), (2, 1), (3, 2)]
    # 初始化 alphas 并赋值，表示该位置的 y 有没有在 x 中找到匹配的位置
    alphas = torch.ones(max_len)                                                      # 创建长度为 max_len 的全 1 Tensor alphas
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()                 # mapper_base 中第二列的元素不为 -1 则为 True，表示找到了最佳匹配位置, 转为 1.0, 为 -1 则是 False，表示没有找到最佳匹配位置，转为 0.0
                                                                                      # alphas = [1., 0., 1., 1., 1., 1., 1., ... , 1.]
    # 初始化 mapper 并赋值，表示该位置的 y 在 x 中找到的最佳位置
    mapper = torch.zeros(max_len, dtype=torch.int64)                                  # 创建长度为 max_len 的全 0 Tensor mapper
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]                                # mapper_base 中第二列的元素不论是否为 -1 全放入 mapper 的前 mapper_base.shape[0]
                                                                                      # mapper = [0, -1,  1,  2,  0,  0,  0, ... ,  0]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))   # mapper 后面的按 mapper[i] = i - 1 填充
                                                                                      # mapper = [0, -1,  1,  2,  4,  5,  6, ... ,  max_len-1]
    # 初始化 m，深拷贝为 alphas
    m = copy.deepcopy(alphas)

    # 初始化 alpha_e 和 alpha_m
    alpha_e = torch.zeros_like(alphas)
    alpha_m = torch.zeros_like(alphas)
    

    # 对 x 和 y 进行 tokenization
    x = tokenizer(
            x,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
    y = tokenizer(
            y,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

    # 对 x 和 y 进行编码
    x_latent = encoder(x)[0].squeeze(0)
    y_latent = encoder(y)[0].squeeze(0)

    i = 0
    # 在 y_seq 和 x_seq 中查找那些不匹配的段落，在 y_seq 找到其中的一个词代表不匹配段，在 x_seq 找到对应的另一个词代表对应的不匹配段，这两个代表词可以代表两个不匹配段成为最佳匹配
    while i < len(y_seq):
        start = None
        if alphas[i] == 0:                                                                         # 检查当前索引 i 处的 alphas 是否为 0，从不匹配的位置开始
            start = i                                                                              # 为 0 则记录这个位置作为 start
            while alphas[i] == 0:
                i += 1                                                                             # 继续前进直到找到 alphas 不为 0 的位置，即不匹配段的末尾

            # 找到在 start 和 i 之间的 y_seq 与 x_seq 之间的最大相似度，即不匹配段的最大相似度
            # 即使 y 没有匹配的位置也尽量在 x 中找到一个相对匹配的位置
            max_sim = float('-inf')                                                                # 最大相似度初始化为负无穷大
            max_s = None                                                                           # 最大相似度对应的源序列索引初始状态
            max_t = None                                                                           # 最大相似度对应的目标序列索引初始状态
            # 寻找不匹配段中具有最大相似度的元素，索引分别为 max_s 和 max_t，这对元素代表这对不相似段
            for i_target in range(start, i):                                                       # i_target 表示 y_seq 序列中当前正在检查的索引
                for i_source in range(mapper[start - 1] + 1, mapper[i]):                           # i_source 表示 x_seq 序列中当前正在检查的索引
                    sim = F.cosine_similarity(x_latent[i_target], y_latent[i_source], dim=0)       # 计算 x_latent 在 i_source 位置的向量与 y_latent 在 i_target 位置的向量之间的余弦相似度
                    if sim > max_sim:                                                              # 找到最大相似度 max_sim 和对应的 max_s, max_t
                        max_sim = sim
                        max_s = i_source
                        max_t = i_target
            if max_s is not None:
                mapper[max_t] = max_s                                                              # mapper[max_t] 被设置为 max_s，表示在 x_seq 中的最佳匹配位置
                alphas[max_t] = 1                                                                  # alphas[max_t] 被设置为 1，表示找到了匹配的位置
                for t in e_seq:
                  if x_seq[max_s] == t:                                                            # 检查源提示在 max_s 位置的 token 是否出现在目标混合词中。如果出现，则将 alpha_e[max_t] 设置为 1
                    alpha_e[max_t] = 1                                                             # 这表明即使目标提示中没有这个目标混合词 t，但这个词在源提示中存在并且与目标提示的某个分词形成了最佳匹配，可以说目标提示 max_t 位置的词与 t 相同
        i += 1                                                                                     # 找到第一个为 0 的位置


    # 检查 y_seq 中是否存在目标混合词 e_seq 中的 token，并根据匹配情况更新张量 alpha_e
    i = 1
    j = 1
    while (i < len(y_seq) - 1) and (j < len(e_seq) - 1):
        found = True
        while e_seq[j] != y_seq[i]:
            i = i + 1                                                  # 在目标提示中找目标混合词的位置 i
            if i >= len(y_seq) - 1:                                    # 如果超过了目标提示长度，表明没找到
                print("blend word not found!")
                found = False
                break
                raise ValueError("local prompt not found in target prompt")
        if found:
            alpha_e[i] = 1                                             # 找到了位置，则将对应位置 i 的 alpha_m 置 1
        j = j + 1                                                      # 寻找下一个目标混合词

    # 检查 y_seq 中是否存在源混合词 m_seq 中的 token，并根据匹配情况更新张量 alpha_m
    i = 1
    j = 1
    while (i < len(y_seq) - 1) and (j < len(m_seq) - 1):
      while m_seq[j] != y_seq[i]:
        i = i + 1                                                      # 在目标提示中找到源混合词的位置 i
      if m_seq[j] == x_seq[mapper[i]]:                                 # mapper[i] 记录源混合词最佳匹配源混合词位置 i 的位置
        alpha_m[i] = 1                                                 # 如果这个最佳匹配的位置也等于源混合词，则将对应位置 i 的 alpha_m 置 1
        j = j + 1                                                      # 寻找下一个源混合词
      else:
        raise ValueError("mutual prompt not found in target prompt")

    return mapper, alphas, m, alpha_e, alpha_m


# 以 source prompt 为基准，与 target prompt 对比得到相应参数
def get_refinement_mapper(prompts, specifiers, tokenizer, encoder, device, max_len=77):
    x_seq = prompts[0]
    mappers, alphas, ms, alpha_objs, alpha_descs = [], [], [], [], []
    for i in range(1, len(prompts)):
        # 对齐源提示和目标提示，得到 y_seq 对 x_seq 的最佳匹配位置 mapper，是否匹配 alpha
        mapper, alpha, m, alpha_obj, alpha_desc = get_mapper(x_seq, prompts[i], specifiers[i-1], tokenizer, encoder, device, max_len)  
        # 将得到的参数存入列表中
        mappers.append(mapper)
        alphas.append(alpha)
        ms.append(m)
        alpha_objs.append(alpha_obj)
        alpha_descs.append(alpha_desc)
    return torch.stack(mappers), torch.stack(alphas), torch.stack(ms),  torch.stack(alpha_objs), torch.stack(alpha_descs)


# 找出 x_seq 中需要替换的部分，并找到 y_seq 中对应的替换部分。该函数返回两个列表，分别表示需要替换的位置和替换后的位置
def get_replace_inds(x_seq, y_seq, source_replace_seq, target_replace_seq):
    replace_mapper = []                                             # 存储替换后的位置
    replace_alpha = []                                              # 存储需要替换的位置
    source_found = False                                            # 标记是否在 x_seq 中找到了需要替换的部分
    source_match, target_match=[], []                               # source_match 存储 x_seq 中找到的需要替换的部分的位置; target_match: 存储 y_seq 中找到的替换部分的位置
    # 在 x_seq 中查找 source_replace_seq 的匹配部分
    for j in range(len(x_seq)):
        found = True
        for i in range(1, len(source_replace_seq) - 1):
            if x_seq[j + i - 1] != source_replace_seq[i]:
                found = False
                break
        if found:
            source_found = True
            for i in range(1, len(source_replace_seq) - 1):
                source_match.append(j + i - 1)
    # 在 y_seq 中查找 target_replace_seq 的匹配部分
    for j in range(len(y_seq)):
        found = True
        for i in range(1, len(target_replace_seq) - 1):
            if y_seq[j + i - 1] != target_replace_seq[i]:
                found=False
                break
        if found:
            for i in range(1, len(source_replace_seq) - 1):
                target_match.append(j + i - 1)
    if not source_found:
        raise ValueError("replacing object not found in prompt")                         # 如果 source_replace_seq 在 x_seq 中没有找到匹配的部分，则抛出一个 ValueError
    if (len(source_match) != len(target_match)):
        raise ValueError(f"the replacement word number doesn't match for word {i}!")     # 如果找到的需要替换的部分数量与替换部分的数量不一致，则抛出一个 ValueError
    replace_alpha += source_match                                                        # 将找到的需要替换的位置和替换后的位置分别添加到 replace_alpha 和 replace_mapper 中
    replace_mapper += target_match
    return replace_alpha, replace_mapper
    

# 返回 word_place 在 text 经过分词后，其对应的所有分词的索引
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")                                                                                     # 对输入 text 进行分词
    if type(word_place) is str:                                                                                      # word_place 是字符串，则查找该字符串在 split_text 中出现的所有位置，并保存这些位置的索引
        word_place = [i for i, word in enumerate(split_text) if word_place == word]                                  # shape 为 (len(i))  [1, 4, 7]
    elif type(word_place) is int:                                                                                    # word_place 是整数，则将其作为一个列表元素添加到 word_place 中
        word_place = [word_place]                                                                                    # shape 为 (1)       [2,]
    out = []                                                                                                         # 初始化一个空列表 out 来存储单词的索引范围
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]                # 使用分词器 tokenizer 对文本 text 进行编码。这通常会返回一个包含整数的列表，每个整数代表一个标记（token）
                                                                                                                     # 使用 tokenizer.decode([item]) 将整数解码成对应的标记字符串
                                                                                                                     # 使用 .strip("#") 去除标记字符串前面可能存在的 # 字符
                                                                                                                     # 最后的处理结果为分词列表，分词可以为标点，单词，或是两者组合
        cur_len, ptr = 0, 0                                                                                          # 初始化当前单词长度 和 源单词索引

        # 找到 word_place 中的单词经过分词后所有分词的索引
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])                  # 当前单词长度 += 分词长度
            if ptr in word_place:                            # 如果这个源单词索引是 word_place 中的，输出列表加上当前的分词索引
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):              # 如果当前单词长度 >= 源索引单词的长度，说明这个单词处理完毕
                ptr += 1                                     # 指向下一个单词
                cur_len = 0                                  # 清空当前单词长度，进行下一个单词的计算
    return np.array(out)


# 对于长度相同的句子 x, y 找到句子中不同的位置 i, 对 x, y 进行分词，那些不同位置的单词也被分词，返回一个 mapper，他是 x, y 分词后的映射，可以反映那些不同的单词在分词后是如何从 x 映射到 y 的
# mapper 的 shape 为 (vocab_len, vocab_len)
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(' ')                                                                                    # 将 prompt x 分割成单词列表
    words_y = y.split(' ')                                                                                    # 将 prompt y 分割成单词列表
    if len(words_x) != len(words_y):                                                                          # 单词数量必须一样
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]                               # 找到 x 和 y 单词不同的所有位置 i
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]                                      # 得到 i 在 x 分词后的所有索引位置
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]                                      # 得到 i 在 y 分词后的所有索引位置
    mapper = np.zeros((max_len, max_len))                                                                     # 创建 (max_len, max_len)) 的全 0 映射矩阵 mapper
    i = j = 0                                                                                                 # i, j 用于指示 x 和 y 分词后的索引位置
    cur_inds = 0                                                                                              # 指向 inds_replace 列表中的当前索引位置
    while i < max_len and j < max_len:
        # 处理替换位置                                        如果 cur_inds 未超出 inds_source 的长度范围，并且 inds_source 在 cur_inds 位置的第一个索引等于 i，则表示当前处理的是一个需要替换的位置
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:                                     # inds_source 是一个数组列表，元素为那些不同单词的分词列表，每一个列表记录那个单词的分词位置
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]                         # 获取当前替换位置的 inds_source 和 inds_target
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1                                                        # 如果 inds_source 和 inds_target 的长度相同，将它们对应位置的值设为 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio                                                         # 如果长度不同，计算一个比例因子 ratio，并分配给 mapper 中相应的位置
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        # 处理非替换位置                                       如果 cur_inds 仍在 inds_source 的长度范围内，但是当前 inds_source 的第一个索引不等于 i，则表示当前 i 和 j 对应的位置不需要替换
        elif cur_inds < len(inds_source):                                                                     # 不满足 inds_source[cur_inds][0] == i，说明还没到要处理的分词
            mapper[i, j] = 1
            i += 1
            j += 1
        # 处理尾部位置                                         如果 cur_inds 超出了 inds_source 的长度范围，这意味着剩下的都是 x 和 y 中相同的部分
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()


# 以 source prompt 为参照，返回多个 mapper (其实就一个 target prompt)，反映 source prompt 是如何映射到那些 prompts 的
# mappers 的 shape 为 (1, vocab_len, vocab_len)
def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]                                                                 # 将prompts列表中的第一个文本输入赋值给x_seq，作为参考序列
    mappers = []                                                                       # 初始化一个空列表来存储每个prompt的映射结果
    for i in range(1, len(prompts)):                                                   # 遍历 prompts 中的每个 prompt (就遍历一个 target prompt)
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)        # 调用 get_replacement_mapper_() 来获取 source prompt 与 target prompt 的映射关系
        mappers.append(mapper)
    return torch.stack(mappers)