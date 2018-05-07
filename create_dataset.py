import random
from random import randint

random.seed(0)

def classify(num, divider, class_type):
    if class_type == "imbalance":
        res = int(num % divider == 0)
    elif class_type == "balance":
        assert divider % 2 == 1
        mod = num % divider
        if mod == 0:
            res = "Other"
        else:
            res = num % divider % 2
    elif class_type == "multiclass":
        res = num % divider
    elif class_type == "substring":
        res = int(str(divider) in str(num))
    else:
        raise NotImplementedError("unknown class_type")

    return res

def create_data(start, mid, count, max, base, divider, class_type):
    if base != 10:
        raise NotImplementedError("base other than 10 not implemented yet")

    lines = []
    for i in range(start, count):
        if i < mid:
            x = i
        else:
            x = randint(mid, max)
        res = classify(x, divider, class_type)
        if res != "Other":
            lines.append(str(x) + "," + str(res))

    path = "dataset/" + str(base) + "div" + str(divider) + "." + class_type + ".txt"
    file = open(path, "w+")
    output = "\n".join(lines)
    file.write(output)
    file.close

create_data(0, 100000, 200000, 1000000000, 10, 7, "multiclass")
create_data(0, 100000, 200000, 1000000000, 10, 7, "imbalance")
create_data(0, 100000, 200000, 1000000000, 10, 7, "balance")
create_data(0, 100000, 200000, 1000000000, 10, 16, "multiclass")
create_data(0, 100000, 200000, 1000000000, 10, 123, "substring")
