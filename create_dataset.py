file = open("dataset/10div7.v3.txt", "w+")

lines = []
for i in range(10000,99999):
    res = i % 7
    y = res
    line = str(i) + "," + str(y)
    lines.append(line)

output = "\n".join(lines)
file.write(output)

file.close
