import re
import jieba

# 获取模型最大输入长度
max_model_length = 3096  # Qwen2.5-VL的典型限制，需根据实际情况调整

# 预留部分长度给prompt和特殊标记
reserved_length = 512  # 根据prompt长度和模型特性调整
window_size = max_model_length - reserved_length

def sliding_window(text, window_size=window_size, step=window_size//2):
    """滑动窗口生成重叠片段"""
    tokens = jieba.lcut(text)
    num_tokens = len(tokens)
    segments = []
    for i in range(0, num_tokens, step):
        start = i
        end = min(i + window_size, num_tokens)
        segment = ' '.join(tokens[start:end])
        segments.append(segment)
        print(f"第{i//step+1}段: {len(segment)} 字符")
    return segments

def slice_by_punctuation(text, punctuations=r'[。！？；\n]'):
    """按标点符号（如句号、感叹号等）分割文本为段落"""
    segments = re.split(punctuations, text)
    # 过滤空段并去除首尾空格
    return [seg.strip() for seg in segments if seg.strip()]

def slice_by_paragraph(text,punctuations=r'\n+'):
    if not isinstance(text, str):
        raise TypeError("输入必须为字符串类型")
    """按段落分割文本（通过换行符或空行识别自然段落）"""
    # 步骤1：按一个或多个换行符分割文本（匹配段落分隔符）
    segments = re.split(punctuations, text)  # 正则r'\n+'匹配1个或多个连续的换行符
    # 步骤2：过滤空段并清理段落首尾的空格/换行符
    return [seg.strip() for seg in segments if seg.strip()]  # 仅保留非空且清理后的段落

def slice_fixed_length(text, window_size=500):
    """固定长度切片，不重叠"""
    return [text[i:i+window_size] for i in range(0, len(text), window_size) if text.strip()]

def slice_example_en(text,punctuations='\n\n\n'):
    segments=re.split(punctuations,text)
    return [seg.strip() for seg in segments if seg.strip()]

