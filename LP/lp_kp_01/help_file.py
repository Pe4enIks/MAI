out_file = open('fam_pairs.ged', "w")
#цикл по строкам и по символам в строке, если символ ] или [, то мы его не сохраняем в result файл
with open("res_fam.ged") as in_file:
    for line in in_file:
        for symb in line:
            if symb!="[" and symb!="]":
                out_file.write(symb)
out_file.close()
#преобразование из пар ind в пары имен
pairs_list = []
id_name_list = []
with open("fam_pairs.ged") as in_file:
    for line in in_file:
        line = line.strip()
        pairs_list.append(line.split(sep=','))
with open("id_name.ged") as in_file:
    for line in in_file:
        line = line.strip()
        id_name_list.append(line.split(sep=","))
for i in range(len(pairs_list)):
    for j in range(len(pairs_list[i])):
        for k in range(len(id_name_list)):
            if pairs_list[i][j] == id_name_list[k][0]:
                pairs_list[i][j] = id_name_list[k][1]
with open("res.ged", "w") as w_file:
    for pair in pairs_list:
        for i in range(2):
            w_file.write(pair[i])
            if i != 1 : w_file.write(",")
        w_file.write("\n")