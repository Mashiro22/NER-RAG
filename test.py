from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from evaluate import evaluate_result
import torch
from slice_ch import *
from final_process import *

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct")

with open('ner_text/test_en.txt', 'r') as file:
    content_test = file.read()
with open('ner_text/prompt.txt', 'r') as file:
    content_prompt = file.read()

# 定义可用的切片方案
SLICE_METHODS = {
    'sliding_window': sliding_window,    # 现有滑动窗口（带重叠）
    'punctuation': slice_by_punctuation, # 按标点切片
    'paragraph':slice_by_paragraph,
    'fixed_length': slice_fixed_length,   # 固定长度无重叠
}

# 选择切片方案（可通过参数或配置文件切换）
method_name="paragraph"
slice_method = SLICE_METHODS[method_name]  # 切换为按标点切片

# 生成切片
content_win = slice_method(content_test)
print("【切片方式为",method_name,"】")
all_outputs = []

for i, segment in enumerate(content_win):
    print(f"处理第{i+1}/{len(content_win)}段...")
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": segment},
                {"type": "text", "text": content_prompt},
            ],
        }
    ]
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # 生成结果
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    all_outputs.append(output_text[0])
    # 释放显存
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

# 合并结果
all_output = "\n".join(all_outputs)  # 中文直接拼接

# 后处理
final_output=postprocess_text_output(all_output)
print("【pred】")
print(final_output)

# 保存最终结果
with open("ner_text/pred_en.txt", "w", encoding="utf-8") as f:
    f.write(final_output)
evaluate_result()
