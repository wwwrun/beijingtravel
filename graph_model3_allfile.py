import copy
import json
from random import random

import gensim
from graphviz import Digraph
from gensim.models import keyedvectors

from Graph import Node, Edge
import math
import numpy as np

class GraphForEntityV0_4_9():
    def __init__(self,wv_from_text,sentence_dict,all_entity):

        self.sentenceDict=sentence_dict
        self.all_entity=all_entity

        self.Graph = []  # 图中的边集
        self.nodeDict = dict()  # 图中的点集
        self.v_v_Co_Dict = dict()  # 动词之间相似度，查完存于此表中，下一次直接查询该字典
        self.T_T_Sim_Dict = dict()  # 类型之间相似度，查完存于此表中，下一次直接查询该字典
        self.wv_from_text = wv_from_text  # 腾讯预训练语料
        self.triple_list=[] # 获得三元组的列表
        self.entity_type_list=[] # 获得实体类型对


        """
        所有的实体类别
        """
        f = open(r".\model3_file\filtered_types_new.json", "r", encoding="utf-8")  # 经过滤后的选用的实体类别
        json_file_dict = json.load(f)
        self.all_type = list(json_file_dict.keys())
        self.type_order_dict=json_file_dict
        f.close()

        """
        每个实体的所有类别
        """
        f = open(r".\model3_file\new_set_all_entity_classification.json", "r", encoding="utf-8")  # 每个实体根据xlore给出的标签
        file = f.read()
        self.all_entity_type = json.loads(file)
        f.close()

        """
        每个实体的所有类别相似度
        """
        f = open(r".\model3_file\typeSimForFilteredTypes.json", "r", encoding="utf-8")  # 用PMI计算的类型相似度
        file = f.read()
        self.all_type_sim = json.loads(file)
        f.close()

    def get_type_list_order(self):
        entity_type_list=sorted(self.entity_type_list,key=lambda x:x[2],reverse=True)
        return entity_type_list

    # 根据实体类别生成动词与实体的边并加入图中
    def find_edge(self,clean_word,flag,rangeEntityList,seedNode,weight=1):
        n_clean_word = clean_word + flag
        rangeEntityList.append(n_clean_word)
        if n_clean_word not in self.nodeDict:
            entityNode = Node(n_clean_word, 1, 'entity')
            self.nodeDict[n_clean_word] = entityNode
        else:
            entityNode = self.nodeDict[n_clean_word]
        xNode = entityNode
        entityEdge = Edge(seedNode, entityNode, weight)
        if not self.graphHasEdge(self.Graph, entityEdge):  # 判断该边有没有出现在图中
            self.Graph.append(entityEdge)  # 若不在，在图的边集中加入该边
        else:  # 如果该边已经存在，则该边的权重+1
            entityEdge = self.graphHasEdge(self.Graph, entityEdge)  # 从Graph中找到这条边
            entityEdge.weight += 1  # 权重+1
        return xNode

    def getEntityAndRelationBySeeds(self,seed):
        """
        seed: 种子动词的名称
        """
        weight = 1
        rangeEntityList = []  # 初始化实体层tail节点列表
        domainEntityList = []  # 初始化实体层head节点列表
        allEntityList = self.all_entity

        if seed in self.nodeDict:  # 如果种子已经在nodeDict中
            seedNode = self.nodeDict[seed]  # 直接取出seedNode
        else:
            seedNode = Node(seed,1,"relation")  # 否则新建一个seed节点，Node（节点名，score，节点类型），score设为1，节点类型设为"relation"
            self.nodeDict[seed] = seedNode

        for sentID in self.sentenceDict:
            sentence = self.sentenceDict[sentID]  # 遍历每个句子
            for item in sentence:
                word = item[0]
                location = word.find("_")
                Numlocation = word.find("#")
                orig_word = word[Numlocation + 1:location]  # 得到每个item的汉字
                if orig_word == seed:  # 如果该词与seed一样，且是个动词，因为动词里可能有’#‘
                    headNode = None  # 初始化头实体
                    tailNode = None  # 初始化尾实体
                    headFirstVerb = False  # 当在寻找头实体时，遇见了动词
                    tailFirstVerb = False  # 当在寻找尾实体时，遇见了动词

                    # 找尾实体
                    find_n = False
                    find_entity=False
                    find_Ans=False
                    n_clean_word=''
                    for aIndex in range(sentence.index(item) + 1 , len(sentence)):  # 再次遍历在sentence中该动词后的item
                        aItem = sentence[aIndex]
                        aWord = aItem[0]
                        aLocation = aWord.find("_")
                        aNumlocation = aWord.find("#")
                        clean_word = aWord[aNumlocation + 1:aLocation]

                        if '$' in aWord: # 找到标点符号则终止尾实体查找
                            break
                        # 找到两个动词以上则不找尾实体
                        if sentence.index(item)+2<len(sentence):
                            if '_v' in sentence[sentence.index(item)+1][0] and '_v' in sentence[sentence.index(item)+2][0]:
                                tailFirstVerb = True

                        # 如果没找到实体，优先找名词
                        if tailFirstVerb==False and '_n' in aWord and aItem[1]>item[1] and aNumlocation == -1:
                            if aItem[1] - item[1] - len(orig_word) <= 5 and len(clean_word)>1:
                                find_n=True
                                n_clean_word=clean_word
                        # 如果找到方位实体词
                        if tailFirstVerb==False and '_Ans' in aWord and aItem[1]>item[1]:
                            Ans_clean_word=clean_word
                            find_Ans=True

                        # 如果找到实体词
                        if tailFirstVerb == False and (clean_word in allEntityList) and aItem[1] > item[1] and aNumlocation != -1:
                            if len(clean_word)>1 and aItem[1] - item[1] - len(orig_word) <= 5:  # 设置两个实体距离不能超过10
                                find_entity=True
                                entity_clean_word=clean_word
                                break# 只选最近的实体

                        # 如果到最后一个字没有找到这个实体，则将名词作为尾实体
                    if find_Ans is True:
                        print('选择 方位词 tail')
                        tailNode=self.find_edge(Ans_clean_word,'_tail',rangeEntityList,seedNode)
                    if find_entity is True and find_Ans is False:
                        print('选择 实体 tail')
                        tailNode=self.find_edge(entity_clean_word,'_tail',rangeEntityList,seedNode)
                    if find_n is True and find_entity is False and find_Ans is False:
                        print('选择名词实体tail')
                        tailNode=self.find_edge(n_clean_word,'_tail',rangeEntityList,seedNode)

                    # 找头实体
                    find_subject=False
                    find_nh=False
                    find_entity_head=False
                    for aIndex in range(sentence.index(item) - 1,-1,-1):  # 再次遍历在sentence中该动词前的item
                        aItem = sentence[aIndex]
                        aWord = aItem[0]
                        aLocation = aWord.find("_")
                        aNumlocation = aWord.find("#")
                        clean_word = aWord[aNumlocation + 1:aLocation]
                        if '$' in aWord:
                            break
                        if sentence.index(item) - 2>0:
                            if '_v' in sentence[sentence.index(item) - 1][0] and '_v' in sentence[sentence.index(item) - 2][0]:
                                headFirstVerb = True

                        if headFirstVerb is False and '_subject' in aWord:
                            print('找到主语词')
                            find_subject=True
                            subject_clean_word=clean_word

                        if headFirstVerb is False and '_n' in aWord and aNumlocation==-1:
                            if aItem[1] - item[1] - len(orig_word) <= 5 and len(clean_word)>1:
                                find_nh=True
                                nh_clean_word=clean_word

                        if headFirstVerb is False and (clean_word in allEntityList) and aItem[1] < item[1] and aNumlocation != -1:
                            # 设置这个共现实体要在种子的前面 （头实体）
                            if len(clean_word)>1 and item[1] - aItem[1] - len(clean_word) <= 5:  # 设置两个实体距离不能超过10
                                find_entity_head=True
                                entity_head_clean_word=clean_word
                                break # 只选最近的实体

                    if find_subject is True:
                        print('选择subject')
                        headNode=self.find_edge(subject_clean_word,'_head',domainEntityList,seedNode)
                    if find_subject is False and find_entity_head is True:
                        print('选择实体head')
                        headNode=self.find_edge(entity_head_clean_word,'_head',domainEntityList,seedNode)

                    if find_subject is False and find_entity_head is False and find_nh is True:
                        print('选择名词head')
                        headNode=self.find_edge(nh_clean_word,'_head',domainEntityList,seedNode)
                    # 若在含有种子动词节点的一句话中同时含有头实体和尾实体，两者不同
                    if headNode is not None and tailNode is not None and headNode.lemma!=tailNode.lemma:
                        e_e_Edge = Edge(headNode, tailNode, weight)  # 生成头实体与尾实体的边，权重初始化为1
                        if not self.graphHasEdge(self.Graph, e_e_Edge):
                            self.Graph.append(e_e_Edge)  # 若该边不存在，则在图中加入该边
                        else:  # 如果该边已经存在，则该边的权重+1
                            e_e_Edge = self.graphHasEdge(self.Graph, e_e_Edge)  # 从Graph中找到这条边
                            e_e_Edge.weight += 1
                        print('头尾实体相连，边加入图中')
                        triple_string = e_e_Edge.nodeA.lemma+' '+str(seed)+' '+e_e_Edge.nodeB.lemma
                        print('三元组：'+triple_string)
                        if triple_string not in self.triple_list:
                            self.triple_list.append(triple_string)
                        break
        return list(set(domainEntityList)),list(set(rangeEntityList))

    def graphHasEdge(self,graph,edge):
        """
        查找图中是否含有edge边。考虑边的顺序是有序的，且节点的类型要一致
        :param graph:
        :param edge:
        :return:
        """
        node1 = edge.nodeA
        node2 = edge.nodeB
        for i in graph:  # 遍历图中的边集，判断该边的两个实体是否与所要查找的边的两个实体一致
            if node1.lemma == i.nodeA.lemma and node2.lemma == i.nodeB.lemma:
                return i
            if node2.lemma == i.nodeA.lemma and node1.lemma == i.nodeB.lemma:
                return i
        return False

    def getE_A_EdgeWithLemma(self,lemma):
        """
        若给定"relation"节点，查找与此节点相连的entity节点，以及这两个节点构成的边；
        若给定"entity"节点，查找与此节点相连的relation节点，以及这两个节点构成的边；
        lemma：所要查找的词
        """
        edgeList = []
        sum = 0
        for edge in self.Graph:
            if (edge.nodeA.lemma == lemma or edge.nodeB.lemma == lemma) and  \
                    (edge.nodeA.type != edge.nodeB.type) and (edge.nodeA.type in ['entity','relation']) \
                    and (edge.nodeB.type in ['entity','relation']):  # 限制该边的两个节点类型不一致，一个是entity节点，一个是relation节点
                edgeList.append(edge)
                sum += edge.weight  # 计算权重之和
        return edgeList,sum

    def getE_T_EdgeWithLemma(self,lemma):
        """
        若给定"entity"节点，查找与此节点相连的type节点，以及这两个节点构成的边；
        若给定"type"节点，查找与此节点相连的entity节点，以及这两个节点构成的边；
        lemma：所要查找的词
        """
        edgeList = []
        sum = 0
        for edge in self.Graph:
            if (edge.nodeA.lemma == lemma or edge.nodeB.lemma == lemma) and \
                    (edge.nodeA.type != edge.nodeB.type) and edge.nodeA.type in ['entity', 'type'] \
                    and edge.nodeB.type in ['entity', 'type']:  # 限制该边的两个节点类型不一致，一个是entity节点，一个是type节点
                edgeList.append(edge)
                sum += edge.weight
        return edgeList,sum

    def Hypothesis1ForEntity(self,entityNode):
        """
        图强化算法的假设1：实体与关系的相互作用，以及实体与类型的相互作用
        """
        edgeList,sumForEntity = self.getE_A_EdgeWithLemma(entityNode.lemma)  # 获得该实体节点的E（实体）_R（关系）边，以及边的权重之和（根据ei找rj）
        s1ForEntity = 0
        for edge in edgeList:
            relationNode = None
            nodeA = edge.nodeA
            nodeB = edge.nodeB
            if nodeA.type == 'relation':
                relationNode = nodeA  # 找到关系节点
            elif nodeB.type == "relation":
                relationNode = nodeB
            edgeList2,sumForRelation = self.getE_A_EdgeWithLemma(relationNode.lemma)  # 获得该关系节点的R（关系）_E（节点）边，以及边的权重（根据rj找en，以及所有rj_en权重之和）
            s1ForEntity += relationNode.score * (edge.weight/sumForRelation)  # 假设一，实体节点与关系节点的交互：关系节点的score*（该E_R的权重/与该关系相连所有E_R边的权重之和）

        edgeList, sumForEntity = self.getE_T_EdgeWithLemma(entityNode.lemma)  # 获得该实体节点的E（实体）_T（类型）边，以及边的权重之和
        for edge in edgeList:
            typeNode = None
            nodeA = edge.nodeA
            nodeB = edge.nodeB
            if nodeA.type == 'type':
                typeNode = nodeA
            elif nodeB.type == "type":
                typeNode = nodeB  # 获得类型节点
            edgeList2, sumForType = self.getE_T_EdgeWithLemma(typeNode.lemma)  # 获得该类型节点的所有T_E边，以及边的权重之和
            s1ForEntity += typeNode.score * (edge.weight / sumForType)  # 假设一，实体节点与类型节点的交互：类型节点的score*（该E_T边的权重/与该类型相连所有E_T边的权重之和）

        return s1ForEntity

    def Hypothesis1ForRelation(self,relationNode):
        """
        假设1：关系节点只与实体节点进行交互
        """
        edgeList,sumForRelation = self.getE_A_EdgeWithLemma(relationNode.lemma)
        s1ForRelation = 0
        for edge in edgeList:
            entityNode = None
            nodeA = edge.nodeA
            nodeB = edge.nodeB
            if nodeA.type == 'entity':
                entityNode = nodeA
            elif nodeB.type == "entity":
                entityNode = nodeB
            edgeList2,sumForRelation = self.getE_A_EdgeWithLemma(entityNode.lemma)
            s1ForRelation += entityNode.score * (edge.weight/sumForRelation)
        # NewRelationNode = copy.deepcopy(relationNode)
        # NewRelationNode.score = s1ForRelation
        return s1ForRelation

    def Hypothesis1ForType(self,typeNode):
        """
        假设1：类型节点只与实体节点进行交互
        """
        edgeList,sumForType = self.getE_T_EdgeWithLemma(typeNode.lemma)
        s1ForType = 0
        for edge in edgeList:
            entityNode = None
            nodeA = edge.nodeA
            nodeB = edge.nodeB
            if nodeA.type == 'entity':
                entityNode = nodeA
            elif nodeB.type == "entity":
                entityNode = nodeB
            edgeList2,sumForEntity = self.getE_T_EdgeWithLemma(entityNode.lemma)
            s1ForType += entityNode.score * (edge.weight/sumForEntity)
        # NewRelationNode = copy.deepcopy(relationNode)
        # NewRelationNode.score = s1ForRelation
        return s1ForType


    def getE_E_EdgeWithLemma(self,lemma):
        """
        获得与所要查询的节点是同类型节点，以及两个节点之间的边，权重之和
        若给定一个“entity”节点，查找与此节点相连且另一个节点也是“entity”类型的边
        """
        edgeList = []
        sum = 0
        for edge in self.Graph:
            if (edge.nodeA.lemma == lemma or edge.nodeB.lemma == lemma) and (edge.nodeA.type == edge.nodeB.type):
                edgeList.append(edge)
                sum += edge.weight
                # print(edge.nodeA.lemma, edge.nodeB.lemma, edge.weight)
        return edgeList,sum

    def Hypothesis2ForEntity(self,entityNode,beta = 0.1):
        """
        假设2：实体与实体之间的交互
        """
        s2ForEntity = (1-beta) * entityNode.score  # 第一项：保存上一次迭代的结果
        edgeList,sumForEntityi = self.getE_E_EdgeWithLemma(entityNode.lemma)  # 根据ei找ej
        s2ForEntityItem2 = 0
        for edge in edgeList:
            entityJ = None
            nodeA = edge.nodeA
            nodeB = edge.nodeB
            if nodeA.lemma == entityNode.lemma:
                entityJ = nodeB
            elif nodeB.lemma == entityNode.lemma:
                entityJ = nodeA
            edgeListJ ,sumForEntityJ = self.getE_E_EdgeWithLemma(entityJ.lemma)  # 根据ej找en，并获得所有与ej相连的edge的权重之和
            s2ForEntityItem2 += entityJ.score*(edge.weight/sumForEntityJ)
        s2ForEntity += beta * s2ForEntityItem2
        # NewEntityNode = copy.deepcopy(entityNode)
        # NewEntityNode.score = s2ForEntity
        return s2ForEntity

    def Hypothesis2ForRelation(self,relationNode,beta = 0.1):
        """
        假设2：关系与关系之间的交互
        """
        s2ForRelation = (1-beta) * relationNode.score
        edgeList,sumForRelationi = self.getE_E_EdgeWithLemma(relationNode.lemma)  # 根据ai找aj
        s2ForRelationItem2 = 0
        for edge in edgeList:
            RelationJ = None
            nodeA = edge.nodeA
            nodeB = edge.nodeB
            if nodeA.lemma == relationNode.lemma:
                RelationJ = nodeB
            elif nodeB.lemma == relationNode.lemma:
                RelationJ = nodeA
            edgeListJ ,sumForRelationJ = self.getE_E_EdgeWithLemma(RelationJ.lemma)  # 根据aj找an，并获得所有与aj相连的edge的权重之和

            s2ForRelationItem2 += RelationJ.score*(edge.weight/sumForRelationJ)
        s2ForRelation += beta * s2ForRelationItem2
        return s2ForRelation

    def combine1and2(self,node,alpha = 0.2):
        """
        综合假设1和假设2，得到最终修改之后的节点
        """
        if node.type == 'entity':
            s1ForEntity = self.Hypothesis1ForEntity(node)
            s2ForEntity = self.Hypothesis2ForEntity(node)

            # print(node.lemma,s1ForEntity,s2ForEntity)
            newScore = (1-alpha)*s1ForEntity + alpha*s2ForEntity
            newNode = copy.deepcopy(node)
            newNode.score = newScore
            return newNode
        elif node.type == 'relation' :
            s1ForRelation = self.Hypothesis1ForRelation(node)
            s2ForRelation = self.Hypothesis2ForRelation(node)
            # print(node.lemma,s1ForRelation,s2ForRelation)
            newScore = (1-alpha)*s1ForRelation + alpha*s2ForRelation
            newNode = copy.deepcopy(node)
            newNode.score = newScore
            return newNode
        elif node.type == 'type' :
            s1ForType = self.Hypothesis1ForType(node)
            s2ForType = self.Hypothesis2ForRelation(node)  # 因为假设2对于type节点以及relation节点的公式是一样的，所以在这里用计算relation的假设2来计算type的假设2
            # print(node.lemma,s1ForType)
            newScore = (1-alpha)*s1ForType + alpha*s2ForType
            newNode = copy.deepcopy(node)
            newNode.score = newScore
            return newNode

    def getNodeInNodeList(self,lemma,newNodeList):
        """
        根据lemma获得节点
        :param lemma: 需要节点的lemma
        :param newNodeList: 需要查找的列表
        :return:
        """
        for node in newNodeList:
            if node.lemma == lemma:
                return node

    def computeTypeNodeScore(self):
        """
        初始化每一个type节点的score，为与type节点相连的所有entity节点score之和
        """
        for edge in self.Graph:
            if edge.nodeA.type == "type" and edge.nodeB.type == "entity":
                edge.nodeA.score += edge.nodeB.score
            if edge.nodeB.type == "type" and edge.nodeA.type == "entity":
                edge.nodeB.score += edge.nodeA.score

        for edge in self.Graph:
            if edge.nodeA.type == "type" :
                self.nodeDict[edge.nodeA.lemma] = edge.nodeA    # 更新nodeDict中的type节点
            if edge.nodeB.type == "type":
                self.nodeDict[edge.nodeB.lemma] = edge.nodeB

    def generateTypeForEntity(self,nodeList):
        """
        为每一个entity节点找到相应的type节点
        """
        for type in self.all_type:
            locals()["head_" + str(type) + "TypeNode"] = Node(type + "-domain",0,"type")
            locals()["tail_" + str(type) + "TypeNode"] = Node(type + "-range",0,"type")

        # type = "其他"
        # locals()["head_" + str(type) + "TypeNode"] = Node(type + "-domain", 0, "type")
        # locals()["tail_" + str(type) + "TypeNode"] = Node(type + "-range", 0, "type")

        for entityNode in nodeList:
            if entityNode.type == "entity":
                if "head" in entityNode.lemma:
                    hasType = False
                    headStr = entityNode.lemma.split('_')[0]  # 获得头实体的词语
                    if headStr not in self.all_entity_type:  # 如果该词语不在所有实体分类词典中，说明该实体是额外补充的地点实体（地点实体+方位词）
                        headStrAllType = [""]  # 则给该实体补充一个空“”标签，原来为地点
                    else:
                        headStrAllType = self.all_entity_type[headStr]  # 获得该实体的类型标签
                    if headStrAllType==[""]:
                        edge = Edge(entityNode, Node("其他-domain", 0, "type"), 0.5)
                        self.Graph.append(edge)
                        self.entity_type_list.append([edge.nodeA.lemma,edge.nodeB.lemma,10])# 加入实体-类型对
                    else:
                        for type in headStrAllType:
                            if type == "":
                                continue
                            elif type in self.all_type:
                                edge = Edge(entityNode, locals()["head_" + str(type) + "TypeNode"], 1)  # 将该头实体节点与相应的类型节点相连
                                hasType = True
                                self.Graph.append(edge)
                                type_name = edge.nodeB.lemma[:edge.nodeB.lemma.rfind('-')]
                                if [edge.nodeA.lemma, edge.nodeB.lemma,self.type_order_dict[type_name]] not in self.entity_type_list:
                                    self.entity_type_list.append(
                                        [edge.nodeA.lemma, edge.nodeB.lemma, self.type_order_dict[type_name]])
                            #else:



                if "tail" in entityNode.lemma:
                    hasType = False
                    tailStr = entityNode.lemma.split('_')[0]
                    if tailStr not in self.all_entity_type:
                        tailStrAllType = [""]
                    else:
                        tailStrAllType = self.all_entity_type[tailStr]
                    if tailStrAllType==[""]:
                        edge = Edge(entityNode, Node("其他-range", 0, "type"), 0.5)
                        self.Graph.append(edge)
                        self.entity_type_list.append([edge.nodeA.lemma, edge.nodeB.lemma,10])
                    else:
                        for type in tailStrAllType:
                            if type == "":
                                continue
                            elif type in self.all_type:
                                edge = Edge(entityNode,locals()["tail_" + str(type) + "TypeNode"], 1)  # 将该尾实体节点与相应的类型节点相连
                                hasType = True
                                self.Graph.append(edge)
                                type_name=edge.nodeB.lemma[:edge.nodeB.lemma.rfind('-')]
                                if [edge.nodeA.lemma, edge.nodeB.lemma,self.type_order_dict[type_name]] not in self.entity_type_list:
                                    self.entity_type_list.append(
                                        [edge.nodeA.lemma, edge.nodeB.lemma,self.type_order_dict[type_name]])

    def addEdgeBetweenR_R(self,nodeList):
        """
        加入R_R边的权重
        """
        relationNodeList = []  # 获取所有关系节点
        for node in nodeList:
            if node.type == 'relation':
                relationNodeList.append(node)
        for i in range(len(relationNodeList)):
            hNode = relationNodeList[i]  # 头关系节点
            for j in range(i + 1, len(relationNodeList)):
                tNode = relationNodeList[j]  # 尾关系节点
                if (hNode.lemma, tNode.lemma) in self.v_v_Co_Dict:
                    weight = self.v_v_Co_Dict[(hNode.lemma, tNode.lemma)]
                elif (tNode.lemma, hNode.lemma) in self.v_v_Co_Dict:
                    weight = self.v_v_Co_Dict[(tNode.lemma, hNode.lemma)]
                else:
                    if hNode.lemma in self.wv_from_text and tNode.lemma in self.wv_from_text:
                        weight = self.wv_from_text.similarity(hNode.lemma, tNode.lemma)  # 获得余弦相似度
                        self.v_v_Co_Dict[(hNode.lemma, tNode.lemma)] = weight
                    else:
                        weight = 0
                if weight > 0.7:
                    edge = Edge(hNode, tNode, weight)  # 如果相似度>0.7，把边加到Graph中
                    self.Graph.append(edge)
                    self.hasRREdge = True

    def addEdgeBetweenE_E(self,nodeList):
        """
        将尾实体与尾实体之间的边和头实体与头实体之间的边，加入到Graph中
        :param nodeList: Graph所有节点列表
        :return:
        """
        headEntityNodeList = []  # 获取所有头实体节点
        tailEntityNodeList = []  # 获取所有尾实体节点
        for node in nodeList:
            if node.type == 'entity' and "_head" in node.lemma:
                headEntityNodeList.append(node)
            if node.type == 'entity' and "_tail" in node.lemma:
                tailEntityNodeList.append(node)
        """
        计算两头实体之间的相似度
        """
        for i in range(len(headEntityNodeList)):
            hNode = headEntityNodeList[i]  # 头实体节点
            for j in range(i + 1, len(headEntityNodeList)):
                tNode = headEntityNodeList[j]

                hNodeStr = hNode.lemma.split('_')[0]
                tNodeStr = tNode.lemma.split('_')[0]
                if (hNodeStr, tNodeStr) in self.v_v_Co_Dict:
                    weight = self.v_v_Co_Dict[(hNodeStr, tNodeStr)]
                elif (tNodeStr, hNodeStr) in self.v_v_Co_Dict:
                    weight = self.v_v_Co_Dict[(tNodeStr, hNodeStr)]
                else:
                    if hNodeStr in self.wv_from_text and tNodeStr in self.wv_from_text:
                        weight = self.wv_from_text.similarity(hNodeStr, tNodeStr)
                        self.v_v_Co_Dict[(hNodeStr, tNodeStr)] = weight
                    else:
                        weight = 0
                        self.v_v_Co_Dict[(hNodeStr, tNodeStr)] = weight
                if weight > 0.7:
                    edge = Edge(hNode, tNode, weight)
                    if self.graphHasEdge(self.Graph, edge) is False:  # 若该边不在图中
                        # print("Here!!!Be care!!!")
                        self.Graph.append(edge)
        """
       计算两尾实体之间的相似度
       """
        for i in range(len(tailEntityNodeList)):
            hNode = tailEntityNodeList[i]  # 尾实体节点
            for j in range(i + 1, len(tailEntityNodeList)):
                tNode = tailEntityNodeList[j]
                hNodeStr = hNode.lemma.split('_')[0]
                tNodeStr = tNode.lemma.split('_')[0]
                if (hNodeStr, tNodeStr) in self.v_v_Co_Dict:
                    weight = self.v_v_Co_Dict[(hNodeStr, tNodeStr)]
                elif (tNodeStr, hNodeStr) in self.v_v_Co_Dict:
                    weight = self.v_v_Co_Dict[(tNodeStr, hNodeStr)]
                else:
                    if hNodeStr in self.wv_from_text and tNodeStr in self.wv_from_text:
                        weight = self.wv_from_text.similarity(hNodeStr, tNodeStr)
                        self.v_v_Co_Dict[(hNodeStr, tNodeStr)] = weight
                    else:
                        weight = 0
                        self.v_v_Co_Dict[(hNodeStr, tNodeStr)] = weight
                if weight > 0.7:
                    edge = Edge(hNode, tNode, weight)
                    if self.graphHasEdge(self.Graph, edge) is False:
                        # print("Here!!!Be care!!!")
                        self.Graph.append(edge)

    def addEdgeBetweenT_T(self,nodeList):
        """
        将range类型和range类型之间的边，domain类型和domain类型之间的边，加入到Graph中
        :param nodeList: Graph所有节点列表
        :return:
        """
        domainNodeList = []  # 获取所有头实体节点
        rangeNodeList = []  # 获取所有尾实体节点
        for node in nodeList:
            if node.type == 'type' and "-domain" in node.lemma:
                # print(node.lemma)
                domainNodeList.append(node)
            if node.type == 'type' and "-range" in node.lemma:
                rangeNodeList.append(node)
        """
       计算两domain类型之间的相似度
       """
        for i in range(len(domainNodeList)):
            hNode = domainNodeList[i]  # 头实体节点
            for j in range(i + 1, len(domainNodeList)):
                tNode = domainNodeList[j]

                hNodeStr = hNode.lemma.split('-')[0]
                tNodeStr = tNode.lemma.split('-')[0]

                if hNodeStr == '其他' or tNodeStr == '其他':
                    continue
                if (hNodeStr, tNodeStr) in self.T_T_Sim_Dict:
                    weight = self.T_T_Sim_Dict[(hNodeStr, tNodeStr)]
                elif (tNodeStr, hNodeStr) in self.T_T_Sim_Dict:
                    weight = self.T_T_Sim_Dict[(tNodeStr, hNodeStr)]
                else:
                    for item in self.all_type_sim:
                        if item[0] == [hNodeStr, tNodeStr] or item[0] == [tNodeStr, hNodeStr]:
                            weight = item[1]
                            # print("!!!!!!" + str(weight))
                            self.T_T_Sim_Dict[(hNodeStr, tNodeStr)] = weight
                if weight > 2.1:
                    edge = Edge(hNode, tNode, weight)
                    if self.graphHasEdge(self.Graph, edge) is False:
                        # print("Here!!!Be care!!!")
                        self.Graph.append(edge)
        """
       计算两range类型之间的相似度
       """
        for i in range(len(rangeNodeList)):
            hNode = rangeNodeList[i]  # 尾实体节点
            for j in range(i + 1, len(rangeNodeList)):
                tNode = rangeNodeList[j]
                hNodeStr = hNode.lemma.split('-')[0]
                tNodeStr = tNode.lemma.split('-')[0]
                if hNodeStr == '其他' or tNodeStr == '其他':
                    continue
                if (hNodeStr, tNodeStr) in self.T_T_Sim_Dict:
                    weight = self.T_T_Sim_Dict[(hNodeStr, tNodeStr)]
                elif (tNodeStr, hNodeStr) in self.T_T_Sim_Dict:
                    weight = self.T_T_Sim_Dict[(tNodeStr, hNodeStr)]
                else:
                    for item in self.all_type_sim:
                        if item[0] == [hNodeStr, tNodeStr] or item[0] == [tNodeStr, hNodeStr]:
                            weight = item[1]
                            # print("!!!!!!" + str(weight))
                            self.T_T_Sim_Dict[(hNodeStr, tNodeStr)] = weight
                if weight > 2.1:
                    edge = Edge(hNode, tNode, weight)
                    if not self.graphHasEdge(self.Graph, edge):
                        # print("Here!!!Be care!!!")
                        self.Graph.append(edge)


    def normalization(self,typeDict,flag):
        """
        对得到的结果进行归一化
        """
        if flag:
            domainTypeScore = []
            rangeTypeScore = []
            for type in typeDict:
                if 'domain' in type[0]:
                    domainTypeScore.append(type[1])
                if 'range' in type[0]:
                    rangeTypeScore.append(type[1])
            sumForDomain = sum(domainTypeScore)
            sumForRange = sum(rangeTypeScore)
            newDomainTypeDict = dict()
            newRangeTypeDict = dict()
            for type in typeDict:
                if 'domain' in type[0]:
                    newDomainTypeDict[type[0]] = type[1] / sumForDomain
                    #newDomainTypeDict[type[0]] = float(type[1]-np.min(domainTypeScore))/(np.max(domainTypeScore)-np.min(domainTypeScore))
                if 'range' in type[0]:
                    newRangeTypeDict[type[0]] = type[1] / sumForRange
                    #newRangeTypeDict[type[0]] = float(type[1]-np.min(rangeTypeScore))/(np.max(rangeTypeScore)-np.min(rangeTypeScore))

            newDomainTypeDict = sorted(newDomainTypeDict.items(), key=lambda item: item[1], reverse=True)
            newRangeTypeDict = sorted(newRangeTypeDict.items(), key=lambda item: item[1], reverse=True)
            return newDomainTypeDict , newRangeTypeDict
        else:
            Score_list=[]
            new_score_dict={}
            for type in typeDict:
                Score_list.append(type[1])
            for type in typeDict:
                new_score_dict[type[0]]=float(type[1])/(np.sum(Score_list))
            new_score_list=sorted(new_score_dict.items(), key=lambda x: x[1], reverse=True)
            return new_score_list


    def get_result(self):
        nodeList = self.nodeDict.values()
        self.addEdgeBetweenE_E(nodeList)  # 加入实体-实体边。
        self.addEdgeBetweenR_R(nodeList)  # 加入关系-关系边。
        self.generateTypeForEntity(nodeList)  # 加入实体-类型边
        self.computeTypeNodeScore()  # 初始化类型节点score
        self.addEdgeBetweenT_T(nodeList)  # 加入类型-类型边
        print('边加入完毕，开始迭代....')
        max_iter = 2  # 最大迭代次数

        aveModifyVal = []  # 记录每个节点修改值的平均值

        for i in range(max_iter):
            newNodeList = []  # 初始化每一次迭代产生的新的节点列表
            modifyVal = []  # 每个节点修改的值
            for node in nodeList:  # 遍历每个节点
                newNode = self.combine1and2(node)  # 得到一个新的节点，newNode与node的区别就是score不一样
                newNodeList.append(newNode)
                modifyVal.append(abs(newNode.score - node.score))
            aveModifyVal.append(sum(modifyVal) / len(modifyVal))
            for edge in self.Graph:  # 遍历每条边
                edge.nodeA = self.getNodeInNodeList(edge.nodeA.lemma, newNodeList)  # 将每条边的节点用新生成的节点列表中的节点代替
                edge.nodeB = self.getNodeInNodeList(edge.nodeB.lemma, newNodeList)

            nodeList = newNodeList  # 进行下一次迭代
            print('迭代%s ok'% str(i+1))

        relationDict = dict()  # 输出的关系节点
        typeDict = dict()  # 输出的类型节点
        entityDict = dict()  # 输出的实体节点

        for node in nodeList:
            if node.type == 'relation':
                relationDict[node.lemma] = node.score
            elif node.type == 'entity':
                entityDict[node.lemma] = node.score
            elif node.type == 'type':
                typeDict[node.lemma] = node.score

        entityDict = sorted(entityDict.items(), key=lambda item: item[1], reverse=True)  # 给最终的实体节点排序
        relationDict = sorted(relationDict.items(), key=lambda item: item[1], reverse=True)  # 给最终的关系节点根据score排序
        typeDict = sorted(typeDict.items(), key=lambda item: item[1], reverse=True)  # 给最终的类型节点根据score排序
        x = range(len(aveModifyVal))
        y = aveModifyVal
        print('清空图为下一次做准备')
        self.Graph = []
        self.nodeDict = dict()
        # print(nodeScoreDict)
        # 对分值进行归一化
        newDomainTypeDict, newRangeTypeDict = self.normalization(typeDict,True)
        entity_score_list=self.normalization(entityDict,False)
        relation_score_list=self.normalization(relationDict,False)
        return relation_score_list, entity_score_list,newDomainTypeDict,newRangeTypeDict

    def main(self,seed_list):
        """
        主函数
        :param seedList: 种子动词列表
        :return:
        对于每一个种子动词都得到其相关列表
        """
        # 获取每个动词的domain和range分数
        seed_DR_dict={}
        for seed in seed_list:
            self.getEntityAndRelationBySeeds(seed)
            r,e,domain_score,range_score=self.get_result()
            seed_DR_dict[seed]=[domain_score,range_score]
        print('文档domain_range获取完毕......')
        # 获取每个文档的子图
        for seed in seed_list:
            self.getEntityAndRelationBySeeds(seed)  # 获得与种子动词相连的实体
            print('%s 子图构建完毕'% seed)

        # print(self.Graph)
        print('开始处理整个文档的图.........')
        relationDict, entityDict, newDomainTypeDict, newRangeTypeDict=self.get_result()
        graph_score_list = [relationDict,entityDict,newDomainTypeDict,newRangeTypeDict]
        print('文档子图加seed子图处理完毕')
        return graph_score_list,seed_DR_dict


if __name__ == '__main__':
    print("开始加载语料库....")
   # wv_from_text = gensim.models.KeyedVectors.load(
    #    r"D:\\腾讯语料\\Tencent_AILab_ChineseEmbedding\\ChineseEmbedding_allWords.bin", mmap='r')
    wv_from_text=keyedvectors.KeyedVectors.load_word2vec_format(r'C:\tencent_corpus\my_zh_word_emb_new.txt',encoding='utf-8',binary=False)
    #wv_from_text=1
    print('词向量模型加载完毕')

    f = open('./all_entity_12766.json', 'r', encoding="utf-8")  # 所有实体
    file = f.read()
    all_entity = json.loads(file)
    f.close()

    with open(r'./entity_verb_result/subject_ev_all_file_0527.json','r',encoding='utf-8')as f:
    #with open(r'./entity_verb_result/subject_ev_yiheyuan.json', 'r', encoding='utf-8')as f:
        file_dict=json.load(f)
    # 对于每一个文件进行处理，获得动词对应的分数
    file_result_dict={}
    file_score_dict={}
    triple_dict={}
    All_sentence_list=[]
    All_sentence_dict={}
    for key in file_dict:
        verb_list=[]
        name=key
        sentence_dict=file_dict[key]
        # 生成所有句子词典
        for k in sentence_dict:
            All_sentence_list.append(sentence_dict[k])
    All_sentence_dict=dict(enumerate(All_sentence_list))
    print('所有句子字典生成完毕')
    # 由于测试需要暂在一个文档中进行测试，全文档运行时间过长
    for k in All_sentence_dict:
        sentence=All_sentence_dict[k]
        for item in sentence:
            # 在all_entity中加入合并方位的词组
            if 'Ans'in item[0]:
                if item[0][:item[0].rfind('_')] not in all_entity:
                    all_entity.append(item[0][:item[0].rfind('_')])
            # 获取每个文件中的动词即关系,必须带词性的以便后续判断
            if '_v' in item[0] and item[0] not in verb_list:
                verb_list.append(item[0][item[0].find('#')+1:item[0].rfind('_')])
    verb_list=list(set(verb_list))
    print('方位实体及动词列表处理完毕')

    # 开始生成单个文档的图结构
    g1 = GraphForEntityV0_4_9(wv_from_text,All_sentence_dict,all_entity)
    REDR_score_list,seed_DR_dict=g1.main(verb_list)
    #file_score_dict[name]=REDR_score_list
    #entity_type_list=g1.entity_type_list
    entity_type_list=g1.get_type_list_order()
    print('domain、range及文档图处理完毕..................')
    # with open('./entity_type_list_new.json', 'w+', encoding='utf-8')as f:
    #     json.dump(entity_type_list, f, ensure_ascii=False)
    # 获取生成的三元组
    tri_score_dict={}
    for tri in g1.triple_list:
        tri_l=tri.split(' ')
        head_e=tri_l[0]
        relation_e=tri_l[1]
        tail_e=tri_l[2]
        find_head_type=False
        find_tail_type = False
        head_type=''
        tail_type=''
        for item in entity_type_list:
            if find_head_type is False and item[0]==head_e:
                head_type=item[1]
                find_head_type=True
            if find_tail_type is False and item[0]==tail_e:
                tail_type=item[1]
                find_tail_type=True

        # 初始化分数，以防计算错误
        domain_type_score,range_type_score,head_score,tail_score,relation_score=0.01,0.01,0.01,0.01,0.01
        for item in seed_DR_dict[relation_e][0]:
            if item[0] == head_type:
                domain_type_score=item[1]
        for item in seed_DR_dict[relation_e][1]:
            if item[0] == tail_type:
                range_type_score = item[1]

        for item in REDR_score_list[1]:
            if item[0] == head_e:
                head_score =item[1]
            if item[0] == tail_e:
                tail_score=item[1]
        for item in REDR_score_list[0]:
            if item[0] == relation_e:
                relation_score = item[1]

        #tri_score=math.log((domain_type_score*range_type_score+head_score*tail_score)/relation_score + 0.5)
        # tri_score = math.log(
        #     (domain_type_score * range_type_score + relation_score) / (head_score * tail_score) + 0.5)
        a=0.6
        tri_score = math.log(
            a*(domain_type_score * range_type_score + relation_score)+(1-a)*(head_score * tail_score) + 0.5)
        temp_dict={head_type: domain_type_score,tail_type: range_type_score,'relation':relation_score,
                                        'head':head_score,'tail':tail_score}
        tri_score_dict[tri]=[tri_score,temp_dict]
    tri_score_order = sorted(tri_score_dict.items(), key=lambda x: x[1][0], reverse=True)
    #triple_dict[name]=tri_score_order
        #print('one document finish')
    print('All 文档处理完毕')
    with open('./model3_result_ALL迭代2_new.json','w+',encoding='utf-8')as f:
       json.dump(tri_score_order,f,ensure_ascii=False)

