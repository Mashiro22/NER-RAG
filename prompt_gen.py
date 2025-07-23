#  python prompt_gen.py

import dspy
# from dspy.teleprompt import BootstrapFewShot
from dspy import BootstrapFewShot, ChainOfThought, Example
import itertools  # 用于生成参数组合
from final_process import *
from evaluate import evaluate_result
# from get_dataset import custom_trainset as trainset


#  定义并设置大模型
base_url = "http://127.0.0.1:11434"
lm = dspy.LM(model="ollama/mistral", api_base=base_url)
dspy.settings.configure(lm=lm)


# ---------------------- 读取文件内容 ----------------------
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
def generate_prompt(combined_instruction,prompt_style,demos_list,prompt_num=3):
    """生成多个不同的Prompt"""
    prompt_generator = ChainOfThought('question -> answer')
    prompts = []
     # 定义风格与示例的组合（可扩展为动态配对）
    style_example_pairs = [
        (prompt_style['1'], demos_list[0]),  # 风格1+示例集1
        (prompt_style['2'], demos_list[0]),  # 风格2+示例集1
        (prompt_style['1'], demos_list[1])   # 风格1+示例集2
    ]
    for i in range(prompt_num):
        # 可选：可通过调整demos参数生成不同风格的prompt（如使用不同示例组合）
        current_style, current_demos = style_example_pairs[i]
        full_question = f"{combined_instruction}\n{current_style}"
        response = prompt_generator(question=full_question)
        # processed_prompt = postprocess_text_output(response.answer)
        prompts.append(response)
        print(f"生成Prompt {i+1} 完成")
    return prompts
def classify_entities(text, prompt, demos, classify_model):
    """使用特定Prompt进行实体分类"""
    # 构造带Prompt的输入
    full_input = f"""根据以下Prompt对文本进行实体分类：
    Prompt: {prompt}
    文本: {text}
    请输出分类结果"""
    if demos:
        response = classify_model(question=full_input, demos=demos)
    else:
        response = classify_model(question=full_input)
    return postprocess_text_output(response.answer)

if __name__ == "__main__":
    prompt_content = read_file("ner_text/prompt.txt")  # 读取分类规则
    test_content = read_file("ner_text/test_en.txt")     # 读取待分类文本
    combined = """Generate a universal prompt that can guide the model to output the OUPUT part in question based on the INPUT in question, similar to the prompt in answer. Finally, output this prompt."""
    prompt_style={
        '1':"This prompt should be more detailed",
        '2':"this prompt should be more concise"
    }
    # ---------------------- 构建 Few-Shot 示例 ----------------------
    # 根据 prompt 要求构造示例（需包含完整格式的输入输出）
    example_input_1 = """
    Input:
    At the International Conference in Paris, leaders from Google and Facebook discussed AI. John Smith from London attended, along with Maria Garcia from Madrid.
    OUPUT:
    ORG：International Conference, Google, Facebook
    PER：John Smith, Maria Garcia
    LOC：Paris, London, Madrid"""
    example_output_1 = """
    ## Task:  
    Classify entities in the input text into the following three categories and output the results directly:  
    - **ORG** (Organizations/Institutions: companies, governments, conferences, etc.)  
    - **PER** (Person Names: specific individual names only, excluding titles/appellations)  
    - **LOC** (Geographical Locations: cities, countries, regions, etc.)  
    
    ## Format Requirements:  
    1. Output three lines, each starting with the category name (ORG:/PER:/LOC:).  
    2. Separate entities with English commas, no extra spaces.  
    3. Keep only one instance of duplicate entities; remove title prefixes (e.g., "Dr. Smith" → "Smith").  
    
    ## Example Format:  
    ORG: Google, NASA  
    PER: Elon Musk  
    LOC: London, Tokyo  """
    
    example_input_2 = """
    Input:
    At the Global Tech Forum in Singapore, executives from Apple, Microsoft, and Alibaba discussed future innovations. Dr. Emma Johnson, a researcher from Stanford University, presented findings on AI. Delegates from Tokyo and Sydney included Mr. Li Wei and Sarah Thompson, representing their respective organizations.
    OUPUT:
    ORG：Global Tech Forum, Apple, Microsoft, Alibaba, Stanford University  
    PER：Emma Johnson, Li Wei, Sarah Thompson  
    LOC：Singapore, Tokyo, Sydney  """
    example_output_2 = """
    Classify entities in the input text into the following three categories and output the results directly:  
    - **ORG** (Organizations/Institutions: companies, governments, conferences, etc.)  
    - **PER** (Person Names: specific individual names only, excluding titles/appellations)  
    - **LOC** (Geographical Locations: cities, countries, regions, etc.)  
    
    ## Format Requirements:  
    1. Output three lines, each starting with the category name (ORG:/PER:/LOC:).  
    2. Separate entities with English commas, no extra spaces.  
    3. Keep only one instance of duplicate entities; remove title prefixes (e.g., "Dr. Smith" → "Smith").    """
    # 创建示例对象（question 为输入文本，answer 为分类结果）
    example_1 = Example(
        question=example_input_1,
        answer=example_output_1
    )
    example_2 = Example(
        question=example_input_2,
        answer=example_output_2
    )
    demos_list = [
        [example_1],  # 示例集1（简单实体）
        [example_2]   # 示例集2（复杂实体）
    ]
    ##生成并评估多个prompt
    
    # 1. 生成3个不同的Prompt
    generated_prompts = generate_prompt(
        combined_instruction=combined,
        prompt_style=prompt_style,
        demos_list=demos_list,  # 可扩展为多个示例列表实现不同风格
        prompt_num=3
    )
    # 2. 初始化分类模型
    classify_model = ChainOfThought('question -> answer')
    # 3. 基于不同Prompt生成分类结果并评价
    f1_scores = []
    for idx, prompt in enumerate(generated_prompts, 1):
        # 生成分类结果
        result = classify_entities(test_content, prompt, classify_model)
        # 保存结果
        output_path = f"ner_text/pred_en_prompt{idx}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        # 评价结果（假设evaluate_result支持传入文件路径）
        current_f1 = evaluate_result(output_path)
        f1_scores.append(current_f1)
        print(f"Prompt {idx} 分类结果已保存至 {output_path}，F1分数：{current_f1:.4f}")
    # 4. 输出综合评价
    print("\n=== 综合评价 ===")
    print(f"平均F1分数：{sum(f1_scores)/len(f1_scores):.4f}")
    print("各Prompt F1明细：", {f"Prompt{idx+1}": score for idx, score in enumerate(f1_scores)})# 直接输出符合格式的分类结果
    max_F1=max(f1_scores)
    index=0
    for i in range(3):
        if f1_scores[i]==max_F1:
            with open("ner_text/gold_en_prompt.txt", "w", encoding="utf-8") as f:
                f.write(str(generated_prompts[i]))
            print(f"选择第{i+1}个style_example_pairs组合作为最优解")


