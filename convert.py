# python convert.py

import re
from collections import defaultdict
import spacy

# 加载NLP模型用于主题分类（可选）
nlp = spacy.load("en_core_web_sm")

def parse_test_temp(file_path):
    sections = []
    current_section = {"text": [], "entities": []}
    current_entity = {"tokens": [], "label": None}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_section["text"]:
                    sections.append(current_section)
                    current_section = {"text": [], "entities": []}
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[1]
            current_section["text"].append(token)
            
            # 合并连续的同标签实体
            if label != 'O':
                if label == current_entity.get("label"):
                    current_entity["tokens"].append(token)
                else:
                    if current_entity["label"]:
                        current_section["entities"].append(current_entity)
                    current_entity = {"tokens": [token], "label": label}
            else:
                if current_entity["label"]:
                    current_section["entities"].append(current_entity)
                    current_entity = {"tokens": [], "label": None}
    
    # 处理最后一个实体
    if current_entity["label"]:
        current_section["entities"].append(current_entity)
    if current_section["text"]:
        sections.append(current_section)
    return sections

def map_label(src_label):
    # 标签映射规则
    if 'person-' in src_label:
        return 'PER'
    elif 'location-' in src_label or 'building-' in src_label:
        return 'LOC'
    elif 'organization-' in src_label:
        return 'ORG'
    elif 'art-' in src_label:
        return 'ART'
    return 'O'

def classify_topic(text):
    # 基于关键词的主题分类（可扩展）
    text = ' '.join(text)
    if 'stadium' in text or 'Games' in text:
        return "SportsEvents"
    elif 'Police' in text or 'law enforcement' in text:
        return "LawEnforcement"
    elif 'album' in text or 'song' in text:
        return "Music"
    elif 'machine guns' in text or 'grenade launchers' in text:
        return "Weapons"
    elif 'Hospital' in text:
        return "Healthcare"
    elif 'Theatre' in text or 'Playhouse' in text:
        return "Theaters"
    elif 'Audi' in text or 'Mercedes-Benz' in text:
        return "Cars"
    else:
        return "General"

def generate_database_format(sections):
    output = []
    for section in sections:
        text = ' '.join(section["text"])
        entities = section["entities"]
        
        # 映射实体标签并分组
        entity_map = defaultdict(list)
        for ent in entities:
            mapped_label = map_label(ent["label"])
            if mapped_label == 'O':
                continue
            entity_text = ' '.join(ent["tokens"])
            entity_map[mapped_label].append(entity_text)
        
        # 生成ORG/PER/LOC字符串
        orgs = ','.join(entity_map['ORG']) if entity_map['ORG'] else ''
        pers = ','.join(entity_map['PER']) if entity_map['PER'] else ''
        locs = ','.join(entity_map['LOC']) if entity_map['LOC'] else ''
        
        # 主题分类
        topic = classify_topic(section["text"])
        
        # 生成输出块
        block = f"##{topic}:\n{text}\n\n"
        block += f"ORG：{orgs}\nPER：{pers}\nLOC：{locs}\n\n\n"
        output.append(block)
    return ''.join(output)

# 主函数
if __name__ == "__main__":
    sections = parse_test_temp("test_1.txt")
    database_output = generate_database_format(sections)
    with open("database.txt", "w", encoding="utf-8") as f:
        f.write(database_output)