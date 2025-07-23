def postprocess_text_output(text_output):
    text_output = text_output.replace('，', ',')  # 统一英文逗号为中文
    text_output = text_output.replace(', ', ',')
    # 假设final_output是模型生成的原始字符串
    lines = text_output.strip().split('\n')
    org_entities = []
    per_entities = []
    loc_entities = []
    
    for line in lines:
        if line.startswith('ORG'):
            org_entities.extend(line[4:].split(','))
        elif line.startswith('PER'):
            per_entities.extend(line[4:].split(','))
        elif line.startswith('LOC'):
            loc_entities.extend(line[4:].split(','))
    
    
    # 将列表转换为集合，便于去重操作
    org_set = set(org_entities)
    per_set = set(per_entities)
    loc_set = set(loc_entities)
    
    # # 从机构（ORG）中剔除同时出现在人名（PER）和地点（LOC）的实体
    # org_set = org_set - per_set - loc_set
    
    # # 从人名（PER）中剔除同时出现在机构（ORG）和地点（LOC）的实体
    # per_set = per_set - org_set - loc_set
    
    # # 从地点（LOC）中剔除同时出现在机构（ORG）和人名（PER）的实体
    # loc_set = loc_set - org_set - per_set
    
    # 转换回列表并保持原有去重逻辑
    org_entities = [ent for ent in org_set if ent.strip() and ent != '无']
    per_entities = [ent for ent in per_set if ent.strip() and ent != '无']
    loc_entities = [ent for ent in loc_set if ent.strip() and ent != '无']
    
    # 重新组装结果
    text_output = (
        f"ORG：{','.join(org_entities)}\n"
        f"PER：{','.join(per_entities)}\n"
        f"LOC：{','.join(loc_entities)}"
    )
    return text_output