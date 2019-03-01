# coding = UTF-8
import random
import math
import numpy as np

pi = 3.1415926535897932384626433832795

L1Flag = True
bern = False
ins_cut = 8.0
sub_cut = 8.0
dataSet = 'YAGO39K'
RAND_MAX = 32767


def rand(min_value, max_value):
    return min_value + (max_value - min_value) * random.randint(0, RAND_MAX) / (RAND_MAX + 1.0)


def normal(x, miu, sigma):
    return 1.0 / math.sqrt(2 * pi) / sigma * math.e ^ (-1 * (x - miu) * (x - miu) / (2 * sigma * sigma))


def rand_normal(miu, sigma, min_value, max_value):
    x = rand(min_value, max_value)
    y = normal(x, miu, sigma)
    d_scope = rand(0.0, normal(miu, miu, sigma))
    while d_scope > y:
        x = rand(min_value, max_value)
        y = normal(x, miu, sigma)
        d_scope = rand(0.0, normal(miu, miu, sigma))
    return x


def sqr(x):
    return x * x


def vec_len(a):
    square = 0
    for i in a:
        square += i * i
    square = math.sqrt(square)
    return square


def norm(a):
    x = vec_len(a)
    b = a
    if x > 1:
        for i in range(len(a)):
            b[i] = a[i] / x
    return b


def norm_r(r):
    if r > 1:
        r = 1
    return r


def rand_max(x):
    remainder = (random.randint(0, RAND_MAX) * random.randint(0, RAND_MAX)) % x
    while remainder < 0:
        remainder += x
    return remainder


relation_num, entity_num, concept_num, triple_num = 0, 0, 0, 0
train_size = 0
rate, margin, margin_instance, margin_subclass = 0.0, 0.0, 0.0, 0.0
res = 0.0
n = 0
concept_instance = list()
instance_concept = list()
instance_brother = list()
sub_up_concept = list()
up_sub_concept = list()
concept_brother = list()
left_entity = dict()
right_entity = dict()
left_num = dict()
right_num = dict()

ok = dict()
sub_class_of_ok = dict()
instance_of_ok = dict()
sub_class_of = list()
instance_of = list()
fb_h = list()
fb_l = list()
fb_r = list()
relation_vec = list()
entity_vec = list()
concept_vec = list()
relation_tmp = list()
entity_tmp = list()
concept_tmp = list()
concept_r = list()
concept_r_tmp = list()


def add_hrt(x, y, z):
    fb_h.append(x)
    fb_l.append(y)
    fb_r.append(z)
    ok[(x, z)] = dict()
    ok[(x, z)][y] = 1
    return True


def add_sub_class_of(sub, parent):
    sub_class_of.append((sub, parent))
    sub_class_of_ok[(sub, parent)] = 1
    return True


def add_instance_of(instance, concept):
    instance_of.append((instance, concept))
    instance_of_ok[(instance, concept)] = 1
    return True


def setup(n_in, rate_in, margin_in, margin_ins, margin_sub):
    n = n_in
    rate = rate_in
    margin = margin_in
    margin_instance = margin_ins
    margin_subclass = margin_sub

    for i in range(len(instance_concept)):
        for j in range(len(instance_concept[i])):
            for k in range(len(concept_instance[instance_concept[i][j]])):
                if concept_instance[instance_concept[i][j]][k] != i:
                    instance_brother[i].append(concept_instance[instance_concept[i][j]][k])

    for i in range(len(sub_up_concept)):
        for j in range(len(sub_up_concept[i])):
            for k in range(len(up_sub_concept[sub_up_concept[i][j]])):
                if up_sub_concept[sub_up_concept[i][j]][k] != i:
                    instance_brother[i].append(up_sub_concept[sub_up_concept[i][j]][k])

    relation_vec = np.zeros([relation_num, n], dtype=np.float64)
    entity_vec = np.zeros([entity_num, n], dtype=np.float64)
    relation_tmp = np.zeros([relation_num, n], dtype=np.float64)
    entity_tmp = np.zeros([entity_num, n], dtype=np.float64)
    concept_vec = np.zeros([concept_num, n], dtype=np.float64)
    concept_tmp = np.zeros([concept_num, n], dtype=np.float64)
    concept_r = np.zeros([concept_num], dtype=np.float64)
    concept_r = np.zeros([concept_num], dtype=np.float64)

    for i in range(relation_num):
        for ii in range(n):
            relation_vec[i][ii] = rand_normal(0, 1.0/n, -6/math.sqrt(n), 6/math.sqrt(n))

    for i in range(entity_num):
        for ii in range(n):
            entity_vec[i][ii] = rand_normal(0, 1.0 / n, -6 / math.sqrt(n), 6 / math.sqrt(n))
        entity_vec[i] = norm(entity_vec[i])

    for i in range(concept_num):
        for ii in range(n):
            concept_vec[i][ii] = rand_normal(0, 1.0 / n, -6 / math.sqrt(n), 6 / math.sqrt(n))
        concept_vec[i] = norm(concept_vec[i])

    for i in range(concept_num):
        concept_r[i] = rand(0, 1)

    train_size = len(fb_h) + len(instance_of) + len(sub_class_of)
    return True


def train_hlr(i, cut):
    pr = 500
    if bern:
        pr = 1000 * right_num[fb_r[i]] / (right_num[fb_r[i]] + left_num[fb_r[i]])
    if random.randint(0, RAND_MAX) % 1000 < pr:
        while True:
            if fb_l[i] in instance_brother:
                if random.randint(0, RAND_MAX) % 10 < cut:
                    j = rand_max(entity_num)
                else:
                    j = random.randint(0, RAND_MAX) % int(len(instance_brother[fb_l[i]]))
                    j = instance_brother[fb_l[i]][j]
            else:
                j = rand_max(entity_num)
            if j not in ok[(fb_h[i], fb_r[i])]:
                break
        do_train_hlr(fb_h[i], fb_l[i], fb_r[i], fb_h[i], j, fb_r[i])
    else:
        while True:
            if fb_h[i] in instance_brother:
                if random.randint(0, RAND_MAX) % 10 < cut:
                    j = rand_max(entity_num)
                else:
                    j = random.randint(0, RAND_MAX) % int(len(instance_brother[fb_h[i]]))
                    j = instance_brother[fb_h[i]][j]
            else:
                j = rand_max(entity_num)
            if j not in ok[(fb_h[i], fb_r[i])]:
                break
        do_train_hlr(fb_h[i], fb_l[i], fb_r[i], fb_l[i], j, fb_r[i])
    relation_tmp[fb_r[i]] = norm(relation_tmp[fb_r[i]])
    entity_tmp[fb_h[i]] = norm(entity_tmp[fb_h[i]])
    entity_tmp[fb_l[i]] = norm(entity_tmp[fb_h[i]])
    entity_tmp[j] = norm(entity_tmp[j])
    return True


def train_instance_of(i, cut):
    i = i - len(fb_h)
    j = 0
    if random.randint(0, RAND_MAX) % 2 == 0:
        while True:
            if instance_of[i][0] in instance_brother:
                if random.randint(0, RAND_MAX) % 10 < cut:
                    j = rand_max(entity_num)
                else:
                    j = random.randint(0, RAND_MAX) % int(len(instance_brother[instance_of[i][0]]))
                    j = instance_brother[instance_of[i][0]][j]
            else:
                j = rand_max(entity_num)
            if (j, instance_of[i][1]) not in instance_of_ok:
                break
        do_train_instance_of(instance_of[i][0], instance_of[i][1], j, instance_of[i][1])
        entity_tmp[j] = norm(entity_tmp[j])
    else:
        while True:
            if instance_of[i][1] in concept_brother:
                if random.randint(0, RAND_MAX) % 10 < cut:
                    j = rand_max(concept_num)
                else:
                    j = random.randint(0, RAND_MAX) % int(len(concept_brother[instance_of[i][1]]))
                    j = concept_brother[instance_of[i][1]][j]
            else:
                j = rand_max(concept_num)
            if (instance_of[i][0], j) not in instance_of_ok:
                break
        do_train_instance_of(instance_of[i][0], instance_of[i][1], instance_of[i][0], j)
        concept_tmp[j] = norm(concept_tmp[j])
        concept_r_tmp[j] = norm_r(concept_r_tmp[j])
    entity_tmp[instance_of[i][0]] = norm(entity_tmp[instance_of[i][0]])
    concept_tmp[instance_of[i][1]] = norm(concept_tmp[instance_of[i][1]])
    concept_r_tmp[instance_of[i][1]] = norm_r(concept_r_tmp[instance_of[i][1]])


def train_sub_class_of(i, cut):
    i = i - len(fb_h) - len(instance_of)
    j = 0
    if random.randint(0, RAND_MAX) % 2 == 0:
        while True:
            if sub_class_of[i][0] in concept_brother:
                if random.randint(0, RAND_MAX) % 10 < cut:
                    j = rand_max(concept_num)
                else:
                    j = random.randint(0, RAND_MAX) % int(len(concept_brother[sub_class_of[i][0]]))
                    j = concept_brother[sub_class_of[i][0]][j]
            else:
                j = rand_max(concept_num)
            if (j, sub_class_of[i][1]) not in sub_class_of_ok:
                break
        do_train_sub_class_of(sub_class_of[i][0], sub_class_of[i][1], j, sub_class_of[i][1])
    else:
        while True:
            if sub_class_of[i][1] in concept_brother:
                if random.randint(0, RAND_MAX) % 10 < cut:
                    j = rand_max(concept_num)
                else:
                    j = random.randint(0, RAND_MAX) % int(len(concept_brother[sub_class_of[i][1]]))
                    j = concept_brother[sub_class_of[i][1]][j]
            else:
                j = rand_max(concept_num)
            if (j, sub_class_of[i][0]) not in sub_class_of_ok:
                break
        do_train_sub_class_of(sub_class_of[i][0], sub_class_of[i][1], sub_class_of[i][0], j)
    concept_tmp[sub_class_of[i][0]] = norm(concept_tmp[sub_class_of[i][0]])
    concept_tmp[sub_class_of[i][1]] = norm(concept_tmp[sub_class_of[i][1]])
    concept_tmp[j] = norm(concept_tmp[j])
    concept_r_tmp[sub_class_of[i][0]] = norm_r(concept_r_tmp[sub_class_of[i][0]])
    concept_r_tmp[sub_class_of[i][1]] = norm_r(concept_r_tmp[sub_class_of[i][1]])
    concept_r_tmp[j] = norm_r(concept_r_tmp[j])
    return True


def do_train_hlr(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b):
    sum1 = calc_sum_hlt(e1_a, e2_a, rel_a)
    sum2 = calc_sum_hlt(e1_b, e2_b, rel_b)
    if sum1 + margin > sum2:
        res = res + margin + sum1 - sum2  # 为什么呀
        gradient_hlr(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b)
    return True


def do_train_instance_of(e_a, c_a, e_b, c_b):
    sum1 = calc_sum_instance_of(e_a, c_a)
    sum2 = calc_sum_instance_of(e_b, c_b)
    if sum1 + margin_instance > sum2:
        res += margin_instance + sum1 - sum2
        gradient_instance_of(e_a, c_a, e_b, c_b)
    return True


def do_train_sub_class_of(c1_a, c2_a, c1_b, c2_b):
    sum1 = calc_sum_sub_class_of(c1_a, c2_a)
    sum2 = calc_sum_sub_class_of(c1_b, c2_b)
    if sum1 + margin_subclass > sum2:
        res += margin_subclass + sum1 - sum2
        gradient_sub_class_of(c1_a, c2_a, c1_b, c2_b)
    return True


def calc_sum_hlt(e1, e2, rel):
    sum_value = 0
    if L1Flag:
        for ii in range(n):  # n需要注意一下
            sum_value += math.fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii])
    else:
        for ii in range(n):  # n需要注意一下
            sum_value += sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii])
    return sum_value


def calc_sum_instance_of(e, c):
    dis = 0
    for i in range(n):
        dis += sqr(entity_vec[e][i] - concept_vec[e][i])
    if dis < sqr(concept_r[c]):
        return 0
    return dis - sqr(concept_r[c])


def calc_sum_sub_class_of(c1, c2):
    dis = 0
    for i in range(n):
        dis += sqr(concept_vec[c1][i] - concept_vec[c2][i])
    if math.sqrt(dis) < math.fabs(concept_r[c1] - concept_r[c2]):
        return 0
    return dis - sqr(concept_r[c2]) - sqr(concept_r[c1])


def gradient_hlr(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b):
    for ii in range(n):
        x = 2 * (entity_vec[e2_a][ii] - entity_vec[e1_a][ii] - relation_vec[rel_a][ii])
        if L1Flag:
            if x > 0:
                x = 1
            else:
                x = -1
        relation_tmp[rel_a][ii] -= -1 * rate * x
        entity_tmp[e1_a][ii] -= -1 * rate * x
        entity_tmp[e2_a][ii] += -1 * rate * x
        x = 2 * (entity_vec[e2_b][ii] - entity_vec[e1_b][ii] - relation_vec[rel_b][ii])
        if L1Flag:
            if x > 0:
                x = 1
            else:
                x = -1
        relation_tmp[rel_b][ii] -= rate * x
        entity_tmp[e1_b][ii] -= rate * x
        entity_tmp[e2_b][ii] += rate * x
    return True


def gradient_instance_of(e_a, c_a, e_b, c_b):
    dis = 0
    for i in range(n):
        dis += sqr(entity_vec[e_a][i] - concept_vec[c_a][i])
    if dis > sqr(concept_r[c_a]):
        for j in range(n):
            x = 2 * (entity_vec[e_a][j] - concept_vec[c_a][j])
            entity_tmp[e_a][j] -= x * rate
            concept_tmp[c_a][j] -= -1 * x * rate
        concept_r_tmp[c_a] -= -2 * concept_r[c_a] * rate

    dis = 0
    for i in range(n):
        dis += sqr(entity_vec[e_b][i] - concept_vec[c_b][i])
    if dis > sqr(concept_r[c_b]):
        for j in range(n):
            x = 2 * (entity_vec[e_b][j] - concept_vec[c_b][j])
            entity_tmp[e_b][j] -= x * rate
            concept_tmp[c_b][j] -= -1 * x * rate
        concept_r_tmp[c_b] -= -2 * concept_r[c_b] * rate
    return True


def gradient_sub_class_of(c1_a, c2_a, c1_b, c2_b):
    dis = 0
    for i in range(n):
        dis += sqr(concept_vec[c1_a][i] - concept_vec[c2_a][i])
    if math.sqrt(dis) > math.fabs(concept_r[c1_a] - concept_r[c2_a]):
        for i in range(n):
            x = 2 * (concept_vec[c1_a][i] - concept_vec[c2_a][i])
            concept_tmp[c1_a][i] -= x * rate
            concept_tmp[c2_a][i] -= -x * rate
        concept_r_tmp[c1_a] -= 2 * concept_r[c1_a] * rate
        concept_r_tmp[c2_a] -= -2 * concept_r[c2_a] * rate

    dis = 0
    for i in range(n):
        dis += sqr(concept_vec[c1_b][i] - concept_vec[c2_b][i])
    if math.sqrt(dis) > math.fabs(concept_r[c1_b] - concept_r[c2_b]):
        for i in range(n):
            x = 2 * (concept_vec[c1_b][i] - concept_vec[c2_b][i])
            concept_tmp[c1_b][i] -= x * rate
            concept_tmp[c2_b][i] -= -x * rate
        concept_r_tmp[c1_b] -= 2 * concept_r[c1_b] * rate
        concept_r_tmp[c2_b] -= -2 * concept_r[c2_b] * rate
    return True


def do_train():
    nbatches = 100
    nepoch = 1000
    batch_size = train_size/nbatches
    for epoch in range(nepoch):
        res = 0
        for batch in range(nbatches):
            relation_tmp = relation_vec
            entity_tmp = entity_vec
            concept_tmp = concept_vec
            concept_r_tmp = concept_r
            for k in range(train_size):
                i = rand_max(train_size)
                if i < len(fb_r):
                    cut = 10 - int(epoch * 8.0 / nepoch)
                    train_hlr(i, cut)
                elif i < len(fb_r) + len(instance_of):
                    cut = 10 - int(epoch * ins_cut / nepoch)
                    train_instance_of(i, cut)
                else:
                    cut = 10 - int(epoch * sub_cut / nepoch)
                    train_sub_class_of(i, cut)
            relation_vec = relation_tmp
            entity_vec = entity_tmp
            concept_vec = concept_tmp
            concept_r = concept_r_tmp

        if epoch % 1 == 0:
            print('epoch:{},{}'.format(epoch, res))

        if epoch % 500 == 0 or epoch == nepoch -1:
            f2 = open("../vector/" + dataSet + "/relation2vec.vec", 'w', encoding='utf-8')
            f3 = open("../vector/" + dataSet + "/entity2vec.vec", 'w', encoding='utf-8')
            f4 = open("../vector/" + dataSet + "/concept2vec.vec", 'w', encoding='utf-8')

            for i in range(relation_num):
                for ii in range(n):
                    ii_tmp = str(round(relation_vec[i][ii], 6))
                    f2.write(ii_tmp + '\t')
                f2.write('\n')

            for i in range(entity_num):
                for ii in range(n):
                    ii_tmp = str(round(entity_vec[i][ii], 6))
                    f3.write(ii_tmp + '\t')
                f3.write('\n')

            for i in range(concept_num):
                for ii in range(n):
                    ii_tmp = str(round(concept_vec[i][ii], 6))
                    f4.write(ii_tmp + '\t')
                f4.write('\n')
                f4.write(concept_r[i] + '\t')
                f4.write('\n')
            f2.close()
            f3.close()
            f4.close()


def prepare():
    f1 = open("../data/" + dataSet + "/Train/instance2id.txt", 'r', encoding='utf-8')
    f2 = open("../data/" + dataSet + "/Train/relation2id.txt", 'r', encoding='utf-8')
    f3 = open("../data/" + dataSet + "/Train/concept2id.txt", 'r', encoding='utf-8')
    f_kb = open("../data/" + dataSet + "/Train/triple2id.txt", 'r', encoding='utf-8')
    entity_num = int(f1.readline().strip('\n'))
    relation_num = int(f2.readline().strip('\n'))
    concept_num = int(f3.readline().strip('\n'))
    triple_num = int(f_kb.readline().strip('\n'))

    while True:
        line_list = f_kb.readline().strip('\n').split('\t')
        if len(line_list) != 3:
            break
        line_list = list(map(int, line_list))
        add_hrt(line_list[0], line_list[1], line_list[2])
        if bern:
            if [line_list[0]] not in left_entity[line_list[2]]:
                left_entity[line_list[2]] = {}
                left_entity[line_list[2]][line_list[0]] = 1
            else:
                left_entity[line_list[2]][line_list[0]] += 1

            if [line_list[1]] not in right_entity[line_list[2]]:
                right_entity[line_list[2]] = {}
                right_entity[line_list[2]][line_list[1]] = 1
            else:
                right_entity[line_list[2]][line_list[1]] += 1
    f1.close()
    f2.close()
    f3.close()
    f_kb.close()
    # concept_instance = np.zeros([concept_num], dtype=np.int)
    # instance_concept = np.zeros([entity_num], dtype=np.int)
    # sub_up_concept = np.zeros([concept_num], dtype=np.int)
    # up_sub_concept = np.zeros([concept_num], dtype=np.int)
    # instance_brother = np.zeros([entity_num], dtype=np.int)
    # concept_brother = np.zeros([concept_num], dtype=np.int)
    concept_instance = [[] for i in range(concept_num)]
    instance_concept = [[] for i in range(entity_num)]
    sub_up_concept = [[] for i in range(concept_num)]
    up_sub_concept = [[] for i in range(concept_num)]
    instance_brother = [[] for i in range(entity_num)]
    concept_brother = [[] for i in range(concept_num)]
    if bern:
        for i in range(relation_num):
            sum1, sum2 = 0, 0
            for it in left_entity[i]:
                sum1 += 1
                sum2 += left_entity[i][it]
            left_num[i] = sum2 // sum1
            
        for i in range(relation_num):
            sum1, sum2 = 0, 0
            for it in right_entity[i]:
                sum1 += 1
                sum2 += right_entity[i][it]
            right_num[i] = sum2 // sum1

    instance_of_file = open("../data/" + dataSet + "/Train/instanceOf2id.txt", 'r', encoding='utf-8')
    sub_class_of_file = open("../data/" + dataSet + "/Train/subClassOf2id.txt", 'r', encoding='utf-8')
    while True:
        line_list = instance_of_file.readline().strip('\n').split('\t')
        if len(line_list) != 2:
            break
        line_list = list(map(int, line_list))
        add_instance_of(line_list[0], line_list[1])
        instance_concept[line_list[0]].append(line_list[1])  # some problems
        concept_instance[line_list[1]].append(line_list[0])

    while True:
        line_list = sub_class_of_file.readline().strip('\n').split('\t')
        if len(line_list) != 2:
            break
        line_list = list(map(int, line_list))
        add_sub_class_of(line_list[0], line_list[1])
        sub_up_concept[line_list[0]].append(line_list[1])
        up_sub_concept[line_list[1]].append(line_list[0])
    return True
