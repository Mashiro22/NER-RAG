# python spite_tag_text.py

import re

def parse_file(input_path):
    sections = []
    current_section = {"category": None, "text": [], "entities": {"ORG": [], "PER": [], "LOC": []}}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测分类标题
            if line.startswith('##'):
                if current_section["category"] is not None:
                    sections.append(current_section)
                    current_section = {"category": None, "text": [], "entities": {"ORG": [], "PER": [], "LOC": []}}
                category = re.match(r'##([^:]+):', line).group(1)
                current_section["category"] = category
                continue
            
            # 检测实体行
            if line.startswith(('ORG：', 'PER：', 'LOC：')):
                entity_type = line[:3].replace('：', '')
                entities = line[3:].strip().split('，') if line[3:].strip() else []
                current_section["entities"][entity_type] = entities
                continue
            
            # 收集文段内容
            if line and current_section["category"] is not None:
                current_section["text"].append(line)
        
        # 添加最后一个section
        if current_section["category"] is not None:
            sections.append(current_section)
    
    return sections

def save_output(sections):
    paragraphs = []
    categories = []
    
    for section in sections:
        # 生成文段内容
        paragraph = ' '.join(section["text"])
        paragraphs.append(paragraph)
        
        # 生成分类结果
        entities_str = []
        for key in ['ORG', 'PER', 'LOC']:
           # 直接拼接，无论实体是否存在，确保单冒号且保留空值
            entities = section["entities"][key]
            entities_str.append(f"{key}{'，'.join(entities)}")  # 单冒号，空列表时输出空字符串
        category_result = '\n'.join(entities_str)
        categories.append(category_result)
    
    # 写入文件
    with open('paragraphs_final.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n\n'.join(paragraphs))
    
    with open('categories_final.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n\n'.join(categories))

if __name__ == '__main__':
    input_file = 'database_F.txt'
    parsed_data = parse_file(input_file)
    save_output(parsed_data)
    
    print(f"生成文件：")
    print(f"- paragraphs.txt（共{len(parsed_data)}条文段）")
    print(f"- categories.txt（共{len(parsed_data)}条分类结果）")