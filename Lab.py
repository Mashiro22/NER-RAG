# python Lab.py

import dspy
import chromadb
from slice_ch import *
from sentence_transformers import SentenceTransformer
from evaluate import evaluate_result
from dspy import BootstrapFewShot, Example, ChainOfThought
from prompt_gen import classify_entities
import evaluate
from statistics import mean

# 加载嵌入模型
embedder = SentenceTransformer('all-MiniLM-L6-v2')

#  定义并设置大模型
base_url = "http://127.0.0.1:11434"
lm = dspy.LM(model="ollama/mistral", api_base=base_url, temperature=0.2)
dspy.settings.configure(lm=lm)

# 预处理文档片段（假设已切片为 segments）
with open('ner_task/database.txt', 'r') as file:
    content_test = file.read()
segments=slice_example_en(content_test)
document_embeddings = embedder.encode(segments)


# 初始化 Chroma 客户端
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag_collection")

# 添加文档片段和向量到数据库
for i, seg in enumerate(segments):
    collection.add(
        ids=[f"seg_{i}"],
        documents=[seg],
        embeddings=document_embeddings[i].tolist()
    )

# 将用户问题转换为向量，与文档向量进行相似度计算
def semantic_retrieve(query, collection, top_k):
    # 生成查询向量
    query_embedding = embedder.encode([query])[0]
    # 在向量数据库中检索相似文档
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    # 返回匹配的文档片段
    return results['documents'][0]


## 操作区域

# 定义读入函数
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 输入用户问题
classify_model = ChainOfThought('question -> answer')
input_path="ner_task/input/paragraphs_final.txt"
initial_output_path="ner_task/output/initial_output.txt"
final_output_path="ner_task/output/final_output.txt"
gold_path="ner_task/input/categories_final.txt"
prompt_path="ner_text/prompt.txt"
text=read_file(input_path)
gold=read_file(gold_path)
prompt=read_file(prompt_path)

text_slices=re.split('\n\n\n',text)
gold_slices=re.split('\n\n\n',gold)
initial_result=[]
final_result=[]
F1_initial=[]
F1_final=[]

top_k=7

ORG_F1=[]
PER_F1=[]
LOC_F1=[]


for i,text_slice in enumerate(text_slices):
    temp=classify_entities(text_slice, prompt, None, classify_model)
    temp=re.sub(r'\(.*?\)', '', temp)
    temp=temp.replace('N/A','')
    temp=temp.replace('None','')
    initial_result.append(temp)
    print(f"【gold】\n{gold_slices[i]}")
    print(f"【pred_initial】\n{temp}")
    print(f"无RAG处理第{i}/650个文段")
    F1_temp, _=evaluate_result(temp,gold_slices[i])
    F1_initial.append(F1_temp)
    

    relevant_segs = semantic_retrieve(temp, collection, top_k)
    
    temp_slices=re.split('\n',temp)
    
    example=[]
    # print(relevant_segs)
    count_rele=0
    for relevant_seg in relevant_segs:
        segment_slice=re.split('\n\n',relevant_seg)
        
        answer_slices=re.split('\n',segment_slice[1])
        flag=0
        for j,answer_slice in enumerate(answer_slices):
            a=re.split('：',answer_slice)
            b=re.split('：',temp_slices[j])
            # print(f"【b1】\n{b[1]}")
            # print(f"【a1】\n{a[1]}")
            if j>top_k/2+1:
                if b[1] and a[1].strip()=='':
                    flag=1
            # print(flag)
        if flag==1:
            continue
        count_rele+=1
        
        example.append(Example(
            question=segment_slice[0],
            answer=segment_slice[1]
        ))
        # print(f"【QUESTION】：\n{segment_slice[0]}\n【ANSWER】：\n{segment_slice[1]}")
    print(count_rele)
    progress_result=classify_entities(text_slice, prompt, example, classify_model)
    progress_result=re.sub(r'\(.*?\)', '', progress_result)
    progress_result=progress_result.replace('N/A','')
    progress_result=progress_result.replace('None','')
    final_result.append(progress_result)
    print(f"【pred_final】\n{progress_result}")
    print(f"RAG={top_k},处理第{i}/695个文段")
    F1_temp, F1_dict=evaluate_result(progress_result,gold_slices[i])
    ORG_F1.append(F1_dict['ORG'])
    PER_F1.append(F1_dict['PER'])
    LOC_F1.append(F1_dict['LOC'])
    
    F1_final.append(F1_temp)
    print("\n\n")

ORG_F1_avg=mean(ORG_F1)
print(ORG_F1_avg)
PER_F1_avg=mean(PER_F1)
print(PER_F1_avg)
LOC_F1_avg=mean(LOC_F1)
print(LOC_F1_avg)


F1_avg_initial=mean(F1_initial)
print(f"【F1_avg_initial,无RAG】:{F1_avg_initial}")

F1_avg_final=mean(F1_final)
print(f"【F1_avg_final,top_k={top_k}】:{F1_avg_final}")


initial_results = "\n\n\n".join(initial_result)  # 中文直接拼接
final_results = "\n\n\n".join(final_result)

with open(initial_output_path, "w", encoding="utf-8") as f:
    f.write(initial_results)

with open(final_output_path, "w", encoding="utf-8") as f:
    f.write(final_results)

# evaluate_result(initial_output_path,gold_path)
# user_question=initial_result


# # 语义检索相关文档片段
# relevant_segs = semantic_retrieve(user_question, collection, top_k=3)
# # context = "\n\n\n".join(relevant_segs)  # 拼接上下文
# # segments=re.split('\n\n\n',context)
# example=[]
# for segment in relevant_segs:
#     segment_slice=re.split('\n\n',segment)
#     example.append(Example(
#         question=segment_slice[0],
#         answer=segment_slice[1]
#     ))
#     print(f"【QUESTION】：\n{segment_slice[0]}\n【ANSWER】：\n{segment_slice[1]}")

# # # example = Example(
# # #     question=segments[0],
# # #     answer=segments[1]
# # # )

# # 使用 RAG 找到的文段作为 Fewshot 评估结果
# progress_result=classify_entities(text, prompt, example, classify_model)
# with open(final_output_path, "w", encoding="utf-8") as f:
#     f.write(progress_result)
# evaluate_result(final_output_path,gold_path)

# # 构建 RAG 示例（包含上下文的 Few-Shot 样本）
# example_question = "Which university is Jane Wilson affiliated with?"
# example_context = "Dr. Jane Wilson from Stanford University presented research on astrophysics."
# example_answer = "Stanford University"
# rag_example = Example(
#     question=example_question,
#     context=example_context,
#     answer=example_answer
# )







