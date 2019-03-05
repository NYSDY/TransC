# coding = UTF-8
import random
import math
import numpy as np
import time
# import ptvsd
#
# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('10.108.168.167', 5678), redirect_output=True)
#
# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()

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
    return 1.0 / math.sqrt(2 * pi) / sigma * math.exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma))


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


class Train:
    def __init__(self, n, rate, margin, margin_instance, margin_subclass, relation_num, entity_num, concept_num,
                 concept_instance, instance_concept, instance_brother, sub_up_concept, up_sub_concept, concept_brother,
                 left_num, right_num):
        self.n = n
        self.rate = rate
        self.margin = margin
        self.margin_instance = margin_instance
        self.margin_subclass = margin_subclass
        self.res = 0
        self.train_size = 0

        self.relation_num = relation_num
        self.entity_num = entity_num
        self.concept_num = concept_num
        self.concept_instance = concept_instance
        self.instance_concept = instance_concept
        self.instance_brother = instance_brother
        self.sub_up_concept = sub_up_concept
        self.up_sub_concept = up_sub_concept
        self.concept_brother = concept_brother
        self.left_num = left_num
        self.right_num = right_num

        self.ok = dict()
        self.sub_class_of_ok = dict()
        self.instance_of_ok = dict()
        self.sub_class_of = list()
        self.instance_of = list()
        self.fb_h = list()
        self.fb_l = list()
        self.fb_r = list()
        self.relation_vec = list()
        self.entity_vec = list()
        self.concept_vec = list()
        self.relation_tmp = list()
        self.entity_tmp = list()
        self.concept_tmp = list()
        self.concept_r = list()
        self.concept_r_tmp = list()

    def add_hrt(self, x, y, z):
        self.fb_h.append(x)
        self.fb_l.append(y)
        self.fb_r.append(z)
        self.ok[(x, z)] = dict()
        self.ok[(x, z)][y] = 1
        return True

    def add_sub_class_of(self, sub, parent):
        self.sub_class_of.append((sub, parent))
        self.sub_class_of_ok[(sub, parent)] = 1
        self.sub_up_concept[sub].append(parent)
        self.up_sub_concept[parent].append(sub)
        return True

    def add_instance_of(self, instance, concept):
        self.instance_of.append((instance, concept))
        self.instance_of_ok[(instance, concept)] = 1
        self.instance_concept[instance].append(concept)
        self.concept_instance[concept].append(instance)
        return True

    def train_hlr(self, i, cut):
        pr = 500
        if bern:
            pr = 1000 * self.right_num[self.fb_r[i]] / (self.right_num[self.fb_r[i]] + self.left_num[self.fb_r[i]])
        if random.randint(0, RAND_MAX) % 1000 < pr:
            while True:
                if self.fb_l[i] in self.instance_brother:
                    if random.randint(0, RAND_MAX) % 10 < cut:
                        j = rand_max(self.entity_num)
                    else:
                        j = random.randint(0, RAND_MAX) % int(len(self.instance_brother[self.fb_l[i]]))
                        j = self.instance_brother[self.fb_l[i]][j]
                else:
                    j = rand_max(self.entity_num)
                if j not in self.ok[(self.fb_h[i], self.fb_r[i])]:
                    break
            self.do_train_hlr(self.fb_h[i], self.fb_l[i], self.fb_r[i], self.fb_h[i], j, self.fb_r[i])
        else:
            while True:
                if self.fb_h[i] in self.instance_brother:
                    if random.randint(0, RAND_MAX) % 10 < cut:
                        j = rand_max(self.entity_num)
                    else:
                        j = random.randint(0, RAND_MAX) % int(len(self.instance_brother[self.fb_h[i]]))
                        j = self.instance_brother[self.fb_h[i]][j]
                else:
                    j = rand_max(self.entity_num)
                if j not in self.ok[(self.fb_h[i], self.fb_r[i])]:
                    break
            self.do_train_hlr(self.fb_h[i], self.fb_l[i], self.fb_r[i], self.fb_l[i], j, self.fb_r[i])
        self.relation_tmp[self.fb_r[i]] = norm(self.relation_tmp[self.fb_r[i]])
        self.entity_tmp[self.fb_h[i]] = norm(self.entity_tmp[self.fb_h[i]])
        self.entity_tmp[self.fb_l[i]] = norm(self.entity_tmp[self.fb_h[i]])
        self.entity_tmp[j] = norm(self.entity_tmp[j])
        return True

    def train_instance_of(self, i, cut):
        i = i - len(self.fb_h)
        if random.randint(0, RAND_MAX) % 2 == 0:
            while True:
                if self.instance_of[i][0] in self.instance_brother:
                    if random.randint(0, RAND_MAX) % 10 < cut:
                        j = rand_max(self.entity_num)
                    else:
                        j = random.randint(0, RAND_MAX) % int(len(self.instance_brother[self.instance_of[i][0]]))
                        j = self.instance_brother[self.instance_of[i][0]][j]
                else:
                    j = rand_max(self.entity_num)
                if (j, self.instance_of[i][1]) not in self.instance_of_ok:
                    break
            self.do_train_instance_of(self.instance_of[i][0], self.instance_of[i][1], j, self.instance_of[i][1])
            self.entity_tmp[j] = norm(self.entity_tmp[j])
        else:
            while True:
                if self.instance_of[i][1] in self.concept_brother:
                    if random.randint(0, RAND_MAX) % 10 < cut:
                        j = rand_max(self.concept_num)
                    else:
                        j = random.randint(0, RAND_MAX) % int(len(self.concept_brother[self.instance_of[i][1]]))
                        j = self.concept_brother[self.instance_of[i][1]][j]
                else:
                    j = rand_max(self.concept_num)
                if (self.instance_of[i][0], j) not in self.instance_of_ok:
                    break
            self.do_train_instance_of(self.instance_of[i][0], self.instance_of[i][1], self.instance_of[i][0], j)
            self.concept_tmp[j] = norm(self.concept_tmp[j])
            self.concept_r_tmp[j] = norm_r(self.concept_r_tmp[j])
        self.entity_tmp[self.instance_of[i][0]] = norm(self.entity_tmp[self.instance_of[i][0]])
        self.concept_tmp[self.instance_of[i][1]] = norm(self.concept_tmp[self.instance_of[i][1]])
        self.concept_r_tmp[self.instance_of[i][1]] = norm_r(self.concept_r_tmp[self.instance_of[i][1]])

    def train_sub_class_of(self, i, cut):
        i = i - len(self.fb_h) - len(self.instance_of)
        if random.randint(0, RAND_MAX) % 2 == 0:
            while True:
                if self.sub_class_of[i][0] in self.concept_brother:
                    if random.randint(0, RAND_MAX) % 10 < cut:
                        j = rand_max(self.concept_num)
                    else:
                        j = random.randint(0, RAND_MAX) % int(len(self.concept_brother[self.sub_class_of[i][0]]))
                        j = self.concept_brother[self.sub_class_of[i][0]][j]
                else:
                    j = rand_max(self.concept_num)
                if (j, self.sub_class_of[i][1]) not in self.sub_class_of_ok:
                    break
            self.do_train_sub_class_of(self.sub_class_of[i][0], self.sub_class_of[i][1], j, self.sub_class_of[i][1])
        else:
            while True:
                if self.sub_class_of[i][1] in self.concept_brother:
                    if random.randint(0, RAND_MAX) % 10 < cut:
                        j = rand_max(self.concept_num)
                    else:
                        j = random.randint(0, RAND_MAX) % int(len(self.concept_brother[self.sub_class_of[i][1]]))
                        j = self.concept_brother[self.sub_class_of[i][1]][j]
                else:
                    j = rand_max(self.concept_num)
                if (j, self.sub_class_of[i][0]) not in self.sub_class_of_ok:
                    break
            self.do_train_sub_class_of(self.sub_class_of[i][0], self.sub_class_of[i][1], self.sub_class_of[i][0], j)
        self.concept_tmp[self.sub_class_of[i][0]] = norm(self.concept_tmp[self.sub_class_of[i][0]])
        self.concept_tmp[self.sub_class_of[i][1]] = norm(self.concept_tmp[self.sub_class_of[i][1]])
        self.concept_tmp[j] = norm(self.concept_tmp[j])
        self.concept_r_tmp[self.sub_class_of[i][0]] = norm_r(self.concept_r_tmp[self.sub_class_of[i][0]])
        self.concept_r_tmp[self.sub_class_of[i][1]] = norm_r(self.concept_r_tmp[self.sub_class_of[i][1]])
        self.concept_r_tmp[j] = norm_r(self.concept_r_tmp[j])
        return True

    def do_train_hlr(self, e1_a, e2_a, rel_a, e1_b, e2_b, rel_b):
        sum1 = self.calc_sum_hlt(e1_a, e2_a, rel_a)
        sum2 = self.calc_sum_hlt(e1_b, e2_b, rel_b)
        if sum1 + self.margin > sum2:
            self.res = self.res + self.margin + sum1 - sum2
            self.gradient_hlr(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b)
        return True

    def do_train_instance_of(self, e_a, c_a, e_b, c_b):
        sum1 = self.calc_sum_instance_of(e_a, c_a)
        sum2 = self.calc_sum_instance_of(e_b, c_b)
        if sum1 + self.margin_instance > sum2:
            self.res += self.margin_instance + sum1 - sum2
            self.gradient_instance_of(e_a, c_a, e_b, c_b)
        return True

    def do_train_sub_class_of(self, c1_a, c2_a, c1_b, c2_b):
        sum1 = self.calc_sum_sub_class_of(c1_a, c2_a)
        sum2 = self.calc_sum_sub_class_of(c1_b, c2_b)
        if sum1 + self.margin_subclass > sum2:
            self.res += self.margin_subclass + sum1 - sum2
            self.gradient_sub_class_of(c1_a, c2_a, c1_b, c2_b)
        return True

    def calc_sum_hlt(self, e1, e2, rel):
        sum_value = 0
        if L1Flag:
            for ii in range(self.n):  # n需要注意一下
                sum_value += math.fabs(self.entity_vec[e2][ii] - self.entity_vec[e1][ii] - self.relation_vec[rel][ii])
        else:
            for ii in range(self.n):  # n需要注意一下
                sum_value += sqr(self.entity_vec[e2][ii] - self.entity_vec[e1][ii] - self.relation_vec[rel][ii])
        return sum_value

    def calc_sum_instance_of(self, e, c):
        dis = 0
        for i in range(self.n):
            dis += sqr(self.entity_vec[e][i] - self.concept_vec[e][i])
        if dis < sqr(self.concept_r[c]):
            return 0
        return dis - sqr(self.concept_r[c])

    def calc_sum_sub_class_of(self, c1, c2):
        dis = 0
        for i in range(self.n):
            dis += sqr(self.concept_vec[c1][i] - self.concept_vec[c2][i])
        if math.sqrt(dis) < math.fabs(self.concept_r[c1] - self.concept_r[c2]):
            return 0
        return dis - sqr(self.concept_r[c2]) - sqr(self.concept_r[c1])

    def gradient_hlr(self, e1_a, e2_a, rel_a, e1_b, e2_b, rel_b):
        for ii in range(self.n):
            x = 2 * (self.entity_vec[e2_a][ii] - self.entity_vec[e1_a][ii] - self.relation_vec[rel_a][ii])
            if L1Flag:
                if x > 0:
                    x = 1
                else:
                    x = -1
            self.relation_tmp[rel_a][ii] -= -1 * self.rate * x
            self.entity_tmp[e1_a][ii] -= -1 * self.rate * x
            self.entity_tmp[e2_a][ii] += -1 * self.rate * x
            x = 2 * (self.entity_vec[e2_b][ii] - self.entity_vec[e1_b][ii] - self.relation_vec[rel_b][ii])
            if L1Flag:
                if x > 0:
                    x = 1
                else:
                    x = -1
            self.relation_tmp[rel_b][ii] -= self.rate * x
            self.entity_tmp[e1_b][ii] -= self.rate * x
            self.entity_tmp[e2_b][ii] += self.rate * x
        return True

    def gradient_instance_of(self, e_a, c_a, e_b, c_b):
        dis = 0
        for i in range(self.n):
            dis += sqr(self.entity_vec[e_a][i] - self.concept_vec[c_a][i])
        if dis > sqr(self.concept_r[c_a]):
            for j in range(self.n):
                x = 2 * (self.entity_vec[e_a][j] - self.concept_vec[c_a][j])
                self.entity_tmp[e_a][j] -= x * self.rate
                self.concept_tmp[c_a][j] -= -1 * x * self.rate
            self.concept_r_tmp[c_a] -= -2 * self.concept_r[c_a] * self.rate

        dis = 0
        for i in range(self.n):
            dis += sqr(self.entity_vec[e_b][i] - self.concept_vec[c_b][i])
        if dis > sqr(self.concept_r[c_b]):
            for j in range(self.n):
                x = 2 * (self.entity_vec[e_b][j] - self.concept_vec[c_b][j])
                self.entity_tmp[e_b][j] -= x * self.rate
                self.concept_tmp[c_b][j] -= -1 * x * self.rate
            self.concept_r_tmp[c_b] -= -2 * self.concept_r[c_b] * self.rate
        return True

    def gradient_sub_class_of(self, c1_a, c2_a, c1_b, c2_b):
        dis = 0
        for i in range(self.n):
            dis += sqr(self.concept_vec[c1_a][i] - self.concept_vec[c2_a][i])
        if math.sqrt(dis) > math.fabs(self.concept_r[c1_a] - self.concept_r[c2_a]):
            for i in range(self.n):
                x = 2 * (self.concept_vec[c1_a][i] - self.concept_vec[c2_a][i])
                self.concept_tmp[c1_a][i] -= x * self.rate
                self.concept_tmp[c2_a][i] -= -x * self.rate
            self.concept_r_tmp[c1_a] -= 2 * self.concept_r[c1_a] * self.rate
            self.concept_r_tmp[c2_a] -= -2 * self.concept_r[c2_a] * self.rate

        dis = 0
        for i in range(self.n):
            dis += sqr(self.concept_vec[c1_b][i] - self.concept_vec[c2_b][i])
        if math.sqrt(dis) > math.fabs(self.concept_r[c1_b] - self.concept_r[c2_b]):
            for i in range(self.n):
                x = 2 * (self.concept_vec[c1_b][i] - self.concept_vec[c2_b][i])
                self.concept_tmp[c1_b][i] -= x * self.rate
                self.concept_tmp[c2_b][i] -= -x * self.rate
            self.concept_r_tmp[c1_b] -= 2 * self.concept_r[c1_b] * self.rate
            self.concept_r_tmp[c2_b] -= -2 * self.concept_r[c2_b] * self.rate
        return True

    def setup(self):
        for i in range(len(self.instance_concept)):
            for j in range(len(self.instance_concept[i])):
                for k in range(len(self.concept_instance[self.instance_concept[i][j]])):
                    if self.concept_instance[self.instance_concept[i][j]][k] != i:
                        self.instance_brother[i].append(self.concept_instance[self.instance_concept[i][j]][k])

        for i in range(len(self.sub_up_concept)):
            for j in range(len(self.sub_up_concept[i])):
                for k in range(len(self.up_sub_concept[self.sub_up_concept[i][j]])):
                    if self.up_sub_concept[self.sub_up_concept[i][j]][k] != i:
                        self.concept_brother[i].append(self.up_sub_concept[self.sub_up_concept[i][j]][k])

        self.relation_vec = np.zeros([self.relation_num, self.n], dtype=np.float64)
        self.entity_vec = np.zeros([self.entity_num, self.n], dtype=np.float64)
        self.relation_tmp = np.zeros([self.relation_num, self.n], dtype=np.float64)
        self.entity_tmp = np.zeros([self.entity_num, self.n], dtype=np.float64)
        self.concept_vec = np.zeros([self.concept_num, self.n], dtype=np.float64)
        self.concept_tmp = np.zeros([self.concept_num, self.n], dtype=np.float64)
        self.concept_r = np.zeros([self.concept_num], dtype=np.float64)
        self.concept_r = np.zeros([self.concept_num], dtype=np.float64)

        sigma, min_value, max_value = 1.0 / self.n, -6 / math.sqrt(self.n), 6 / math.sqrt(self.n)
        for i in range(self.relation_num):
            for ii in range(self.n):
                self.relation_vec[i][ii] = rand_normal(0, sigma, min_value, max_value)

        for i in range(self.entity_num):
            for ii in range(self.n):
                self.entity_vec[i][ii] = rand_normal(0, sigma, min_value, max_value)
            self.entity_vec[i] = norm(self.entity_vec[i])

        for i in range(self.concept_num):
            for ii in range(self.n):
                self.concept_vec[i][ii] = rand_normal(0, sigma, min_value, max_value)
            self.concept_vec[i] = norm(self.concept_vec[i])

        for i in range(self.concept_num):
            self.concept_r[i] = rand(0, 1)

        self.train_size = len(self.fb_h) + len(self.instance_of) + len(self.sub_class_of)
        return True

    def do_train(self):
        nbatches = 100
        nepoch = 1000
        batch_size = self.train_size // nbatches
        for epoch in range(nepoch):
            self.res = 0
            cut1 = 10 - int(epoch * 8.0 / nepoch)
            cut2 = 10 - int(epoch * ins_cut / nepoch)
            cut3 = 10 - int(epoch * sub_cut / nepoch)
            for batch in range(nbatches):
                self.relation_tmp = self.relation_vec
                self.entity_tmp = self.entity_vec
                self.concept_tmp = self.concept_vec
                self.concept_r_tmp = self.concept_r
                for k in range(batch_size):
                    i = rand_max(self.train_size)
                    if i < len(self.fb_r):
                        # cut = 10 - int(epoch * 8.0 / nepoch)
                        self.train_hlr(i, cut1)
                    elif i < len(self.fb_r) + len(self.instance_of):
                        # cut = 10 - int(epoch * ins_cut / nepoch)
                        self.train_instance_of(i, cut2)
                    else:
                        # cut = 10 - int(epoch * sub_cut / nepoch)
                        self.train_sub_class_of(i, cut3)
                self.relation_vec = self.relation_tmp
                self.entity_vec = self.entity_tmp
                self.concept_vec = self.concept_tmp
                self.concept_r = self.concept_r_tmp

            if epoch % 1 == 0:
                print('epoch:{},{},\t,time:{}'.format(epoch, self.res, time.asctime(time.localtime(time.time()))))

            if epoch % 500 == 0 or epoch == nepoch - 1:
                f2 = open("../vector/" + dataSet + "/relation2vec.vec", 'w', encoding='utf-8')
                f3 = open("../vector/" + dataSet + "/entity2vec.vec", 'w', encoding='utf-8')
                f4 = open("../vector/" + dataSet + "/concept2vec.vec", 'w', encoding='utf-8')

                for i in range(self.relation_num):
                    for ii in range(self.n):
                        ii_tmp = str(round(self.relation_vec[i][ii], 6))
                        f2.write(ii_tmp + '\t')
                    f2.write('\n')

                for i in range(self.entity_num):
                    for ii in range(self.n):
                        ii_tmp = str(round(self.entity_vec[i][ii], 6))
                        f3.write(ii_tmp + '\t')
                    f3.write('\n')

                for i in range(self.concept_num):
                    for ii in range(self.n):
                        ii_tmp = str(round(self.concept_vec[i][ii], 6))
                        f4.write(ii_tmp + '\t')
                    f4.write('\n')
                    f4.write(str(self.concept_r[i]) + '\t')
                    f4.write('\n')
                f2.close()
                f3.close()
                f4.close()


def prepare(n, rate, margin, margin_instance, margin_subclass):
    left_entity = dict()
    right_entity = dict()
    left_num = dict()
    right_num = dict()

    f1 = open("../data/" + dataSet + "/Train/instance2id.txt", 'r', encoding='utf-8')
    f2 = open("../data/" + dataSet + "/Train/relation2id.txt", 'r', encoding='utf-8')
    f3 = open("../data/" + dataSet + "/Train/concept2id.txt", 'r', encoding='utf-8')
    f_kb = open("../data/" + dataSet + "/Train/triple2id.txt", 'r', encoding='utf-8')
    entity_num = int(f1.readline().strip('\n'))
    relation_num = int(f2.readline().strip('\n'))
    concept_num = int(f3.readline().strip('\n'))
    triple_num = int(f_kb.readline().strip('\n'))

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

    train = Train(n, rate, margin, margin_instance, margin_subclass, relation_num, entity_num, concept_num,
                  concept_instance, instance_concept, instance_brother, sub_up_concept, up_sub_concept, concept_brother,
                  left_num, right_num)

    while True:
        line_list = f_kb.readline().strip('\n').split(' ')
        if len(line_list) != 3:
            break
        line_list = list(map(int, line_list))
        train.add_hrt(line_list[0], line_list[1], line_list[2])
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

    instance_of_file = open("../data/" + dataSet + "/Train/instanceOf2id.txt", 'r', encoding='utf-8')
    sub_class_of_file = open("../data/" + dataSet + "/Train/subClassOf2id.txt", 'r', encoding='utf-8')
    while True:
        line_list = instance_of_file.readline().strip('\n').split(' ')
        if len(line_list) != 2:
            break
        line_list = list(map(int, line_list))
        train.add_instance_of(line_list[0], line_list[1])
        # instance_concept[line_list[0]].append(line_list[1])  # 这个参数是类里面要用的
        # concept_instance[line_list[1]].append(line_list[0])

    while True:
        line_list = sub_class_of_file.readline().strip('\n').split(' ')
        if len(line_list) != 2:
            break
        line_list = list(map(int, line_list))
        train.add_sub_class_of(line_list[0], line_list[1])
        # sub_up_concept[line_list[0]].append(line_list[1])
        # up_sub_concept[line_list[1]].append(line_list[0])
    return train


if __name__ == '__main__':
    print('start time:{} '.format(time.asctime(time.localtime(time.time()))))
    random.seed(10)
    train_ex = prepare(100, 0.001, 1, 0.4, 0.3)
    print('prepare end time:{} '.format(time.asctime(time.localtime(time.time()))))
    train_ex.setup()
    print('setup end time:{} '.format(time.asctime(time.localtime(time.time()))))
    train_ex.do_train()
