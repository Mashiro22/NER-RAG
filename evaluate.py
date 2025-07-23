def parse_entity_file(f):
    """解析实体文件，返回 {类别: 实体集合}"""
    entity_dict = {'ORG': set(), 'PER': set(), 'LOC': set()}
    # with open(file_path, 'r', encoding='utf-8') as f:
    lines=f.split('\n')
    # print(f)
    for line in lines:
        # print(line)
        line = line.strip()
        # if not line:  # 跳过空行
        #     continue
        # 提取类别（如ORG:）和实体列表
        line.replace(':', '：')
        # if '：' not in line:
        #     continue  # 跳过没有分隔符的行
        category, entities = line.split('：', 1)  # 按第一个冒号分割
        category = category.upper()  # 统一为大写
        # 分割实体并去空格
        entity_list = [ent.strip() for ent in entities.split(',')]
        entity_dict[category] = set(entity_list)
    return entity_dict

def evaluate_ner(gold, pred):
    """评估NER结果，返回各分类及总体指标"""
    # 解析真实标签和预测结果
    gold = parse_entity_file(gold)
    pred = parse_entity_file(pred)

    F1_temp= {}
    metrics = {}
    for category in ['ORG', 'PER', 'LOC']:
        # 提取真实和预测的实体集合
        gold_ents = gold[category]
        pred_ents = pred[category]
        
        tp = len(gold_ents & pred_ents)  # 真正例：两者共有的实体
        fp = len(pred_ents - gold_ents)  # 假正例：预测有但真实没有
        fn = len(gold_ents - pred_ents)  # 假负例：真实有但预测没有
        
        # 计算精确率、召回率、F1
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        
        metrics[category] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
           'recall': recall,
            'f1': f1
        }
        F1_temp[category]=f1
    
    # 计算宏观平均（三类指标的算术平均）
    macro_precision = sum([m['precision'] for m in metrics.values()]) / 3
    macro_recall = sum([m['recall'] for m in metrics.values()]) / 3
    macro_f1 = sum([m['f1'] for m in metrics.values()]) / 3
    metrics['macro_avg'] = {
        'precision': macro_precision,
       'recall': macro_recall,
        'f1': macro_f1
    }
    
    return metrics, F1_temp

def evaluate_result(pred,gold):
    # 示例运行
    # gold_path = gold_path
    # pred_path = output_path
    results, F1_temp = evaluate_ner(gold, pred)
    
    # 打印结果
    for category, stats in results.items():
        if category == 'macro_avg':
            # print(f"【{category} 平均指标】")
            # print(f"  精确率(Precision): {stats['precision']:.4f}")
            # print(f"  召回率(Recall): {stats['recall']:.4f}")
            print(f"  F1值(F1): {stats['f1']:.4f}")
        # else:
            # print(f"【{category} 实体】")
            # print(f"  真正例(TP): {stats['tp']}")
            # print(f"  假正例(FP): {stats['fp']}")
            # print(f"  假负例(FN): {stats['fn']}")
            # print(f"  精确率(Precision): {stats['precision']:.4f}")
            # print(f"  召回率(Recall): {stats['recall']:.4f}")
            # print(f"  F1值(F1): {stats['f1']:.4f}")
    return results['macro_avg']['f1'], F1_temp