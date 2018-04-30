file = open("dataset/10div7.v2.txt", "w+")

lines = []
for i in range(10000,99999):
    res = i % 7
    if res == 0:
        None
    else:
        if res == 1 or res == 3 or res == 5:
            y = 1
        else:
            y = 0

        line = str(i) + "," + str(y)
        lines.append(line)

output = "\n".join(lines)
file.write(output)

file.close
