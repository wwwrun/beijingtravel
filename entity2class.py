import os
import json
import time


def entity_to_class(entity_set_list):
    dict_patch_class={}
    with open(r'./patch_tag.txt','r',encoding='UTF-8')as fp:
        patch_lines_list=fp.readlines()
        print("len of linelist %d"% len(patch_lines_list))
    # 分类实体，人物、地点、建筑、组织机构、行政区划、学校、历史
    # 对xlore分类文件进行预处理得到{词：[类]}字典
    PER_list = []
    LOC_list = []
    ORG_list = []
    SCH_list = []
    HIS_list = []
    others_list=[]
    print('开始构建字典')
    for entity in entity_set_list:
        for line in patch_lines_list:
            patch_key = line.split('\t\t')[0]
            if patch_key == entity or patch_key[:patch_key.rfind('（')]:
                if line.split('\t\t')[1] != '\n':
                    dict_patch_class[entity] = line.strip().split('\t\t')[1].split('::;')
                else:
                    continue
    print('实体类别字典构建完毕')
    for key in dict_patch_class:
        if '人物' in dict_patch_class[key]:
            PER_list.append(key)
        if '地点' in dict_patch_class[key] or '建筑' in dict_patch_class[key] or '行政区划' in dict_patch_class[key]:
            LOC_list.append(key)
        if '组织机构' in dict_patch_class[key]:
            ORG_list.append(key)
        if '学校' in dict_patch_class[key]:
            SCH_list.append(key)
        if '历史' in dict_patch_class[key]:
            HIS_list.append(key)
        else:
            others_list.append(key)
    print("实体分类完成")
    return PER_list,LOC_list,ORG_list,SCH_list,HIS_list,others_list


if __name__=="__main__":
    with open(r'./all_matrix_result/bdlink_xlink_entity.json','r',encoding='UTF-8')as f:
        f_content=f.read()
    dict_file=json.loads(f_content)
    all_entity_list=dict_file['all_link_list']
    print("开始处理文件...")
    start=time.process_time()
    per_list,loc_list,org_list,sch_list,his_list,other_list=entity_to_class(all_entity_list)
    dict_list={}
    dict_list["per_list"]=per_list
    dict_list["loc_list"]=loc_list
    dict_list["org_list"]=org_list
    dict_list["sch_list"]=sch_list
    dict_list["his_list"]=his_list
    dict_list["others"]=other_list
    end=time.process_time()
    print('字典构建完毕！用时 %s'% str(end-start))
    with open(r'./all_matrix_result/entity2class.json','w+',encoding='utf-8')as f1:
        json.dump(dict_list,f1,ensure_ascii=False)
    print('ok')