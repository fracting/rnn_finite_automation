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
    else:
        raise NotImplementedError("unknown class_type")

    return res

def create_data(start, end, base, divider, class_type):
    if base != 10:
        raise NotImplementedError("base other than 10 not implemented yet")

    lines = []
    for i in range(start, end):
        res = classify(i, divider, class_type)
        if res != "Other":
            lines.append(str(i) + "," + str(res))

    path = "dataset/" + str(base) + "div" + str(divider) + "." + class_type + ".txt"
    file = open(path, "w+")
    output = "\n".join(lines)
    file.write(output)
    file.close

create_data(0, 100000, 10, 7, "imbalance")
create_data(10000, 100000, 10, 7, "balance")
create_data(10000, 100000, 10, 7, "multiclass")

create_data(10000, 100000, 10, 11, "imbalance")
create_data(10000, 100000, 10, 11, "balance")
create_data(10000, 100000, 10, 11, "multiclass")

create_data(10000, 100000, 10, 5, "multiclass")
create_data(10000, 100000, 10, 8, "multiclass")
create_data(0, 100000, 10, 16, "multiclass")
