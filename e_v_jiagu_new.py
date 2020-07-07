# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 2020

@author: www
"""
import json
import re
import os
import time
import jiagu


# jiagu对实体词语进行词性标注,修正标注错误动词情况
def jiagu_process(words):
    #word_list = jiagu.seg(words)
    word_list=[]
    word_list.append(words)
    pos_list = jiagu.pos(word_list)
    word_string_list = list(zip(word_list, pos_list))
    for word in word_string_list:
        if len(word[0])>2 and word[1]=='v':
            #print('标注出现错误')
            word_string=word[0]+'_ne'
            return word_string
    word_string = ' '.join(['_'.join(c) for c in word_string_list])
    return word_string


def invert_dict(d):

    return dict([(v,k) for (k,v) in d.iteritems()])

class my_entity_verb:
    def not_empty(self, s):
        # 去除句中空格及句首句尾空格
        s.strip().replace(' ','')
        return s
       # return s and "".join(s.split())

    def splitSentence(self, text):
        pattern = r'。|！|？|；|='
        result_list = re.split(pattern, text)
        result_list = list(filter(self.not_empty, result_list))
        #    print(result_list)
        return result_list

    def stopwordsList(self):
        """
        return:停用词列表
        """
        stop_list_Path = '.\\中文停用词表.txt'
        f = open(stop_list_Path, 'r', encoding='utf-8')
        stopwords = [line.strip() for line in f.readlines()]
        #    print(stopwords)
        return stopwords

    def splitWordOneSentence(self, sentence):
        stopwords = self.stopwordsList()
        sentence = sentence.strip()
        result_words = []
        word_list= jiagu.seg(sentence)
        pos_list=jiagu.pos(word_list)
        words=list(zip(word_list,pos_list))
        for word in words:
            # 排除数字错标情况,jiagu会标错数字,不确定的就标成动词
            if word[0].isdigit():
                word=(word[0],'m')
            if word[0] not in stopwords and len(word[0].strip()) != 0:
                result_words.append(word)
        return result_words


    def mapEntity(self, sentence_list, all_entity):
        #stopwords=self.stopwordsList()
        entity_all_sentence_list = []
        for sentence in sentence_list:
            entity_in_each_sentence =[]
            flag = False
            entity_verb_dict = {}
            entity_temp = ""
            location_temp = 999
            for entity in all_entity:
                if entity and entity!='编辑':
                    # if entity in sentence:
                    location = sentence.find(entity)
                    if location != -1:
                        flag = True
                        entity_in_each_sentence.append(entity)
                        location_1 = 0
                        index = 0
                        # 遇到相同的短实体则进行下一个实体，处理序列由长实体到短实体
                        if entity in entity_temp and location == location_temp:
                            continue
                        #print('开始进入循环')
                        while location_1 < len(sentence):  # 循环发现find,不要用while循环了，太坑了
                            location_1 = sentence.find(entity, location_1)
                            #print(sentence,location_1)
                            if location_1 == -1:  # 未发现实体下一个实体
                                #print('break')
                                break
                            else:
                                index += 1
                                # entity_verb_dict[str(index) + '#' + entity +'_ne'] = location_1
                                entity_verb_dict[str(index)+ '#' +jiagu_process(entity)] = location_1
                                location_1 += len(entity)
                        # entity_in_each_sentence.append(entity+'_'+str(sentence.find(entity)))
                        entity_temp = entity
                        location_temp = location
                    else:
                        continue
            # 实体处理完毕！
            # 在命中实体情况下加动词、数词、时间词、方位词,排除句子中已经有的实体
            if flag:
                pattern=re.compile(r'(\d{4}年\d{1,2}月\d{1,2}日)|(\d{4}年\d{1,2}月)|(\d{4}年)|(\d{1,2}月)|(\d{1,2}月\d{1,2}日)')
                # m=pattern.search(sentence)
                m=pattern.finditer(sentence)
                if m:
                    for x in m:
                        entity_verb_dict[x.group() + '_nt'] = sentence.find(x.group())

                splitWordList = self.splitWordOneSentence(sentence)

                index1 = 0
                loc = 0
                for word in splitWordList:
                    # 加入符号分割符，方便后续处理
                    if word[0]=='，'or word[0]=="、":
                        loc=sentence.find(word[0],loc)
                        entity_verb_dict[str(index1)+"$"+word[0] + '_' + str(word[1])]=loc
                        index1+=1
                        loc+=1
                    # 加入名词与动词
                    #if word[0] not in entity_in_each_sentence and ('n' in word[1] or 'v' in word[1]):
                    #上句话会删除一些重复的实体，造成句子缺失
                    if word[0] and ('n' in word[1] or 'v' in word[1] or 'q' in word[1] or 'm' in word[1]):
                        entity_verb_dict[word[0] + '_' + str(word[1])] = sentence.find(word[0])
            entity_verb_list = sorted(entity_verb_dict.items(), key=lambda item: item[1])
            # 如果位置一样，按照长在前短在后排序
            entity_verb_list.sort(key=lambda x:(x[1],-len(x[0][x[0].find('#')+1:])))
            if entity_verb_list:
                # 去除短重复,从前往后
                tmp_del_list=[]
                print('删除开始', entity_verb_list)
                for i in range(len(entity_verb_list)-1):
                    for j in range(i+1,len(entity_verb_list)):
                        #print(entity_verb_list[i][0],entity_verb_list[i][1])
                        if entity_verb_list[j][0][entity_verb_list[j][0].find('#')+1:entity_verb_list[j][0].rfind('_')]\
                                in entity_verb_list[i][0] and entity_verb_list[j][1] < entity_verb_list[i][1]+len(entity_verb_list[i][0][:entity_verb_list[i][0].rfind('_')]):

                            tmp_del_list.append(entity_verb_list[j])

                entity_verb_list=list(set(entity_verb_list)-set(tmp_del_list))
                entity_verb_list.sort(key=lambda k:k[1])
                print('删除完毕', entity_verb_list)
            if entity_verb_list:
                entity_all_sentence_list.append(entity_verb_list)
        return entity_all_sentence_list

    def ReadMyAllEntity(self):
        file_name = r"C:\Users\wtsin\PycharmProjects\beijingtravel\tfidf_matrix_" \
                    r"generator\matrix_result\bdlink_xlink_entity.json"
        with open(file_name, 'r', encoding='UTF-8')as f:
            file_content = f.read()
        all_entity = json.loads(file_content)['all_link_list']

        return all_entity

    def Getsymbolword(self):
        symbolwords_file=r'./symbol_words_9859.json'
        with open(symbolwords_file,'r',encoding='utf-8')as f:
            symbol_word_list=json.load(f)

        return symbol_word_list

    def GetAllEntity(self):
        entity_path=r"./all_entity_12767.json"
        with open(entity_path,'r',encoding='utf-8')as f:
            all_entity_list=json.load(f)
        return all_entity_list


if __name__ == "__main__":
    # 存取所有文件
    file_dict = {}
    # 读取文件
    entity_verb_process = my_entity_verb()
    # 默认模式+景点词典
    jiagu.load_userdict(r'C:\Users\wtsin\PycharmProjects\beijingtravel\allfilelist.txt')
    path = r".\test"
    file_list = os.listdir(path)
    # 实体列表获取
    all_entity = entity_verb_process.GetAllEntity()
    all_entity.sort(reverse=True)  # 对实体进行降序排序，方便短实体淘汰
    # with open("./all_entity"+"_"+str(len(all_entity))+".json",'w+',encoding='utf-8')as fe:
    #     json.dump(all_entity,fe,ensure_ascii=False)
    for file_name in file_list:
        print("开始处理文件：", file_name)
        start = time.process_time()
        f = open('C:\\Users\\wtsin\\PycharmProjects\\beijingtravel\\test\\' + file_name
                 , 'r', encoding='utf-8')
        file = f.read()
        json_file_dict = json.loads(file)  # 读取json格式
        text = json_file_dict["text"]  # 读取text
        sentence_list = entity_verb_process.splitSentence(text)  # 将text分为句子列表
        """不需要考虑句子必须含有大景点"""
        Entity_in_sentence = entity_verb_process.mapEntity(sentence_list, all_entity)
        # with open("C:\\Users\\wtsin\\PycharmProjects\\beijingtravel\\entity_verb_result\\" +"ev_"+
        #           file_name[:-3] + "json", "w", encoding='utf-8') as json_file:
        sentence_dict = {}
        for i in range(len(Entity_in_sentence)):
            sentence_dict[i] = Entity_in_sentence[i]
            file_dict[file_name[file_name.find('_')+1:file_name.rfind('.')]]=sentence_dict
        #    json.dump(sentence_dict, json_file, ensure_ascii=False)
        end = time.process_time()
        print("用时%s" % str(end - start))
        f.close()
    with open(r".\entity_verb_result\ev_all_file_0611.json", "w", encoding='utf-8') as json_file:
        json.dump(file_dict, json_file, ensure_ascii=False)
