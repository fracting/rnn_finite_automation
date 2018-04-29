file = open("dataset/10div7.txt", "w+")

lines = []
for i in range(10000,99999):
    if i % 7 == 0:
        y = 1
    else:
        y = 0

    line = str(i) + "," + str(y)
    lines.append(line)

output = "\n".join(lines)
file.write(output)

file.close
