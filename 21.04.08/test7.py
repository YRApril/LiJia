# encoding: utf-8
from aJPloy_back import initialPopulation, evaluate, getParents, getVari, getCross
import time
from testData import getData

personNum = 50  # 种群大小
length = 17  # 个体长度
mutationProbability = 0.6  # 变异概率

if __name__ == '__main__':
    theScore = []  # 最佳个体选中次数
    bestPerson = []  # 最佳个体列表
    theBestEval = 0
    iteration = 2  # 最大代数
    data = getData()
    mid = int(data.shape[0] * 0.8)

    for i in range(10):  # 设置跑多少轮，用来查看收敛性的
        population = initialPopulation(personNum, length)
        flag = 0
        timeStart = time.time()
        while flag != iteration:
            print("第", flag + 1, "代")
            evalList = []
            tempPopulation = []
            print(population)

            for person in population:
                evalList.append(evaluate(data, mid, person))
            maxEval = max(evalList)
            print('maxEval=', maxEval)
            theIndex = evalList.index(maxEval)
            tempPopulation.append(population[theIndex])  # 每次迭代时先将上一代最大的个体放到下一代种群中
            print("开始交叉")
            for i in range(personNum):
                # 获得用于交叉的父母
                parentsFaIndex = getParents(evalList)
                parentsFa = population[parentsFaIndex]
                parentsMaIndex = getParents(evalList)
                parentsMa = population[parentsMaIndex]
                child = getCross(parentsFa, parentsMa)

                child = getVari(child)
                tempPopulation.append(child)
            population = tempPopulation
            flag += 1

            evalList = []
            for person in population:
                evalList.append(evaluate(data, mid, person))
            maxEval = max(evalList)
            if theBestEval < maxEval:
                theBestEval = maxEval
            theIndex = evalList.index(maxEval)
            person = population[theIndex]
            if person not in bestPerson:
                bestPerson.append(person)
                theScore.append(1)
            else:
                theScore[bestPerson.index(person)] += 1
        print('duration=', time.time() - timeStart)

    print(theScore)
    print(bestPerson)
    theBestEvalList = []
    for item in bestPerson:
        theBestEvalList.append(evaluate(data, mid, item))
    print(theBestEvalList)
    print(theBestEval)
    print(max(theScore))
