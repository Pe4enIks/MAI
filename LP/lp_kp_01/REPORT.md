# Отчет по курсовому проекту
## по курсу "Логическое программирование"

### студент: Пищик Е.С.

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |               |
| Левинская М.А.|              |      5--     |

> *Комментарии проверяющих (обратите внимание, что более подробные комментарии возможны непосредственно в репозитории по тексту программы)*
Грамматика в 5 п. (местоимения в запросах)
## Введение

Я смог получить опыт программирования на декларативном ЯП, что является однозначно плюсом, ибо заставляет настроить своё мышление немного по-другому, не так как в императивных ЯП. Также я научился правильно описывать предикаты, составлять правила и факты для решения задач, узнал как Prolog ищет свои решения, как происходит откат и отсечение, как работает рекурсия в Prolog и т.д. Научился работать с файлами и циклами в Prolog.

## Задание

 1. Создать родословное дерево своего рода на несколько поколений (3-4) назад в стандартном формате GEDCOM с использованием сервиса MyHeritage.com 
 2. Преобразовать файл в формате GEDCOM в набор утверждений на языке Prolog, используя следующее представление: `child(ребенок, родитель)`, `male(человек)`, `female(человек)`.
 3. Реализовать предикат проверки/поиска тёщи - `mother_in_law(тёща, зять)`.
 4. Реализовать программу на языке Prolog, которая позволит определять степень родства двух произвольных индивидуумов в дереве.
 5. [На оценки хорошо и отлично] Реализовать естественно-языковый интерфейс к системе, позволяющий задавать вопросы относительно степеней родства, и получать осмысленные ответы. 

## Получение родословного дерева

Я получил дерево при помощи сайта MyHeritage.com, количество индивидов в дереве = 23.

## Конвертация родословного дерева

Я использовал два ЯП - Prolog и Python. Prolog я решил использовать, чтобы лучше понять как он устроен и как он работает, Python я использовал в месте, которое не смог реализовать на Prolog.

Все пердикаты входящие в решение:
```prolog
%парсинг по полу
parse_sex :- 
    open('tree.ged', read, In),
    open('sex_terms.ged', write, Out), 
    repeat, 
    read_line_to_string(In, X),
    split_string(X, " ", " ", M_STR), 
    my_write(M_STR, Out),
    nl, 
    X=end_of_file, !, 
    nl, 
    close(In),
    close(Out).
%парсинг fam для женщины
parse_fam_fem:- 
    open('tree.ged', read, In),
    open('fam_terms_fem.ged', write, Out), 
    repeat, 
    read_line_to_string(In, X),
    split_string(X, " ", " ", M_STR), 
    my_write_1(M_STR, Out),
    nl, 
    X=end_of_file, !, 
    nl, 
    close(In),
    close(Out).
%парсинг fam для мужчины
parse_fam_m:- 
    open('tree.ged', read, In),
    open('fam_terms_m.ged', write, Out), 
    repeat, 
    read_line_to_string(In, X),
    split_string(X, " ", " ", M_STR), 
    my_write_2(M_STR, Out),
    nl, 
    X=end_of_file, !, 
    nl, 
    close(In),
    close(Out).
%запись в term-ы female(person) и male(person) пролога
to_terms_sex:-
    open('sex_terms.ged', read, In),
    open('res_sex.pl', write, Out),
    repeat,
    read_line_to_string(In, X),
    split_string(X, ",", ",", M_STR),
    write_to_term(M_STR, Out),
    nl, 
    X=end_of_file, !, 
    nl,  
    close(In), 
    close(Out).
%удалить строки где только ребенок, без родителя
delete :-
    open('fam_terms_fem.ged', read, In_1),
    open('fam_terms_m.ged', read, In_2),
    open('res_fam_term.ged', write, Out),
    repeat,
    read_line_to_string(In_1, X),
    split_string(X, ",", ",", M_STR),
    delete_line(M_STR, Out),
    nl, 
    X=end_of_file, !, 
    nl,
    repeat,
    read_line_to_string(In_2, Y),
    split_string(Y, ",", ",", M_STR_1),
    delete_line(M_STR_1, Out),
    nl, 
    Y=end_of_file, !, 
    nl,    
    close(In_1),
    close(In_2), 
    close(Out).
%запись в term-ы female(person) и male(person) пролога
write_to_term(M_STR, Out) :- member("F", M_STR), nth0(0, M_STR, GIVN), nth0(1, M_STR, SURN) -> 
    write(Out, "female('"), write(Out, GIVN),write(Out, "_"), write(Out, SURN), write(Out, "').\n"); write(Out, "").
write_to_term(M_STR, Out) :- member("M", M_STR), nth0(0, M_STR, GIVN), nth0(1, M_STR, SURN) -> 
    write(Out, "male('"), write(Out, GIVN),write(Out, "_"), write(Out, SURN), write(Out, "').\n"); write(Out, "").
%для записи имя/фамилия/пол
my_write(M_STR, Out) :- 
    member("GIVN", M_STR), member("2", M_STR), nth0(2, M_STR, Elem_1) -> 
    write(Out, Elem_1), write(Out, ","); write(Out, "").
my_write(M_STR, Out) :- 
    member("SURN", M_STR), member("2", M_STR), nth0(2, M_STR, Elem_1) -> 
    write(Out, Elem_1), write(Out, ","); write(Out, "").
my_write(M_STR, Out) :- 
    member("SEX", M_STR), member("1", M_STR), nth0(2, M_STR, Elem_1) -> 
    write(Out, Elem_1), write(Out, "\n"); write(Out, "").
%женщина и дети
my_write_1(M_STR, Out) :- 
    member("FAM", M_STR), member("0", M_STR) -> 
    write(Out, "\n"); write(Out, "").
my_write_1(M_STR, Out) :- 
    member("CHIL", M_STR), member("1", M_STR), nth0(2, M_STR, CHILD) -> 
    write(Out, CHILD), write(Out, ","); write(Out, "").
my_write_1(M_STR, Out) :- 
    member("WIFE", M_STR), member("1", M_STR), nth0(2, M_STR, WIFE) -> 
    write(Out, WIFE), write(Out, ","); write(Out, "").
%мужчина и дети
my_write_2(M_STR, Out) :- 
    member("FAM", M_STR), member("0", M_STR) -> 
    write(Out, "\n"); write(Out, "").
my_write_2(M_STR, Out) :- 
    member("CHIL", M_STR), member("1", M_STR), nth0(2, M_STR, CHILD) -> 
    write(Out, CHILD), write(Out, ","); write(Out, "").
my_write_2(M_STR, Out) :- 
    member("HUSB", M_STR), member("1", M_STR), nth0(2, M_STR, HUSB) -> 
    write(Out, HUSB), write(Out, ","); write(Out, "").
%ничего пока что
my_write_3(M_STR, Out) :- 
    member("GIVN", M_STR), member("2", M_STR), nth0(2, M_STR, Elem_1) -> 
    write(Out, Elem_1), write(Out, "_"); write(Out, "").
my_write_3(M_STR, Out) :- 
    member("SURN", M_STR), member("2", M_STR), nth0(2, M_STR, Elem_1) -> 
    write(Out, Elem_1), write(Out, "\n"); write(Out, "").
my_write_3(M_STR, Out) :- 
    member("INDI", M_STR), member("0", M_STR), nth0(1, M_STR, IND) -> 
    write(Out, IND), write(Out, ","); write(Out, "").
%удалить строки где только ребенок, без родителя
delete_line(M_STR, Out) :- nth0(1, M_STR, El) -> write(Out, M_STR), write(Out, "\n"); write(Out, "").
%является ли Elem членом списка
member(Elem, [Elem|_Tail]).
member(Elem, [_Head|Tail]):- member(Elem, Tail).
%в Elem записывается элемент списка пои индексу, если он существует, иначе fail
nth0(0, [Elem|_Tail], Elem):-!.
nth0(Index, _List, _Elem):-
   Index < 0, !, fail.
nth0(Index, [_Head|Tail], Elem):-
   NextIndex is Index - 1,
   nth0(NextIndex, Tail, Elem).
%получить всех детей для каждого родителя
get_child:-
    open('res_fam_term.ged', read, In),
    open('res_fam.ged', write, Out),
    repeat, 
    read_line_to_string(In, X),
    split_string(X, ",", ",", M_STR),
    get_ch(M_STR, Out, 1),
    nl,
    X=end_of_file, !,
    nl,
    close(In),
    close(Out).
get_ch(M_STR, Out, IND) :- nth0(0, M_STR, PARENT), nth0(IND, M_STR, CHILD) ->
    write(Out, CHILD), write(Out, ","), write(Out, PARENT), write(Out, "\n"),
    get_ch(M_STR, Out, IND+1); writef("").
%получение пар имя=id
get_id_name :-
    open('tree.ged', read, In),
    open('id_name.ged', write, Out),
    repeat, 
    read_line_to_string(In, X),
    split_string(X, " ", " ", M_STR),
    my_write_3(M_STR, Out),
    nl,
    X=end_of_file, !,
    nl,
    close(In),
    close(Out).
%меняем пары с id на term-ы chil(name1, name2)
to_term_fam :-
    open('res.ged', read, In),
    open('res_fam.pl', write, Out),
    repeat, 
    read_line_to_string(In, X),
    split_string(X, ",", ",", M_STR),
    term(M_STR, Out, X),
    nl,
    X=end_of_file, !,
    nl,
    close(In),
    close(Out).
term(M_STR, Out, X) :- X\=end_of_file -> nth0(0, M_STR, CHILD), nth0(1, M_STR, PARENT),
    write(Out, "child('"), write(Out, CHILD), write(Out, "','"), write(Out, PARENT), write(Out, "').\n"); writef("").
%объединяем res файлы в один res файл
res_union :-
    open('res_sex.pl', read, In_1),
    open('res_fam.pl', read, In_2),
    open('res.pl', write, Out),
    repeat,
    read_line_to_string(In_1, X),
    print(Out, X),
    nl,
    X=end_of_file, !,
    nl,
    write(Out, "\n"),
    repeat,
    read_line_to_string(In_2, Y),
    print(Out, Y),
    nl,
    Y=end_of_file, !,
    nl,
    close(In_1),
    close(In_2),
    close(Out).
print(Out, X) :- X=end_of_file -> writef(""); write(Out, X), write(Out, "\n").
```

Чтобы получить финальный файл `res.pl`, сначала нужно в `parser.pl` сделать запрос `step_1`. Сначала парсим пол человека, затем разделяем по полу в разные предикаты, затем записываем всё в один файл.
```prolog
step_1 :-
    parse_sex,
    parse_fam_fem,
    parse_fam_m,
    delete,
    get_child,
    get_id_name,
    to_terms_sex.
```
Следующим шагом нужно запустить `help_file.py`. Убиваем ненужные квадратные скобки, и заменяем индексы на имена.
```python
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
```
Потом снова заходим в `parser.pl` и пишем запрос `step_2`. В котором мы просто записываем в файл предикаты `child(ребенок, отец)` из полученных нами пар файла `res.ged`, полученного на предыдущем шаге. И объединяем результаты полученные в `step_1` - предикаты `male(человек)`, `female(человек)` с новыми предикатами `child(ребенок, отец)`.
```prolog
step_2:-
    to_term_fam,
    res_union.
```
Результатом будет файл `res.pl`.

## Предикат поиска родственника

Предикат `mother_in_law(X,Y)` определяет является ли `X` тёщей `Y`, при помощи предиката `mother(X,Y)` находим для какого `Z`, `X` - мама, потом определяем при помощи предиката `wife(X,Y)` для какого `Y`, `Z` - жена, говорим при помощи `male(X)`, что пол `Y` - мужской.
```prolog
%пункт 3 КП, вариант 6 - тёща
father(X, Y) :- child(Y, X), male(X).
mother(X, Y) :- child(Y, X), female(X).
husband(X, Y) :- child(Z, X), child(Z, Y), male(X), X\=Y.
wife(X, Y) :- child(Z, X), child(Z, Y), female(X), X\=Y.
%является ли X тёщей Y
mother_in_law(X, Y) :- mother(X, Z), wife(Z, Y), male(Y).
```

Пример использования:
```prolog
2 ?- mother_in_law(X, Y).
X = 'Eva_Samara',
Y = 'Sergei_Pishchik' ;
X = 'Eva_Samara',
Y = 'Sergei_Pishchik' ;
X = 'Eva_Samara',
Y = 'Nikolai_Lyachovec' ;
X = 'Eva_Samara',
Y = 'Nikolai_Lyachovec' ;
X = 'Maria_Los',
Y = 'Petr_Samara' ;
X = 'Maria_Los',
Y = 'Petr_Samara' ;
X = 'Maria_Los',
Y = 'Petr_Samara' ;
X = 'Aleksandra_Kozlova',
Y = 'Nikolai_Pishchik' ;
false.
```

## Определение степени родства

Сначала опрделим предикаты для всех родственников, используя предикаты `child(X,Y)`, `male(X)`, `female(X)`. 
```prolog
%пункт 4 КП
son(X, Y) :- child(X, Y), male(X), X\=Y.
daughter(X, Y) :- child(X, Y), female(X), X\=Y.
brother(X, Y) :- child(X, Z), child(Y, Z), male(X), X\=Y.
sister(X, Y) :- child(X, Z), child(Y, Z), female(X),  X\=Y.

%отношения родства
fam(husband, HUSBAND, WIFE) :- husband(HUSBAND, WIFE).
fam(wife, WIFE, HUSBAND) :- wife(WIFE, HUSBAND).
fam(brother, BROTHER, Y) :- brother(BROTHER, Y).
fam(sister, SISTER, Y) :- sister(SISTER, Y).
fam(father, FATHER, CHILD) :- father(FATHER, CHILD).
fam(mother, MOTHER, CHILD) :-  mother(MOTHER, CHILD).
fam(son, CHILD, PARENT) :- son(CHILD, PARENT).
fam(daughter, CHILD, PARENT) :- daughter(CHILD, PARENT).
```
Затем создадим общий предикат `relative(RES, X, Y)` - который принимает `RES` - список родственных отношений и проверяет удовлетворяет ли `X` данной цепочке до `Y`. Также можно определить родство `X` и `Y`. `bfs` - предикат поиска в ширину.
```prolog
perm(X, Y) :- fam(_, X, Y).
prlng([X|T], [Y, X|T]) :- perm(X, Y), not(member(Y, [X|T])).

bfs([[H|T]|_], H, [H|T]).
bfs([H|T], RES, TREE) :- findall(W, prlng(H, W), TREES), append(T, TREES, NEWTREES), !, bfs(NEWTREES, RES, TREE).

find([_], R, R).
find([X,Y|T], R, REL) :- fam(RE, X, Y), find([Y|T], [RE|R], REL).

relative(REL, X, Y) :- bfs([[X]], Y, R), reverse(R, RE), find(RE, [], NEWREL), reverse(NEWREL, REL), N is 0.
```

Примеры использования:
```prolog
9 ?- relative([brother], 'Aleksey_Pishchik', T).
T = 'Evgenii_Pishchik' ;
T = 'Evgenii_Pishchik' ;
T = 'Evgenii_Pishchik' ;
T = 'Evgenii_Pishchik' ;
false.
```

```prolog
10 ?- relative(X, 'Aleksey_Pishchik', 'Sergei_Pishchik').  
X = [son] ;
X = [brother, son] ;
X = [brother, son] ;
X = [brother, son] ;
X = [brother, son] ;
X = [son, wife] ;
X = [son, wife] ;
X = [son, wife] ;
X = [son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [brother, son, wife] ;
X = [son, mother, son] ;
false.
```

## Естественно-языковый интерфейс
Определим предикат `question(L)`, который принимает список вида `[is, name_0, relationship, name_1, ?]`, `[whose, relationship, is, name_1, ?]` и выводит на экран ответ. Предикат `word(X)` определяет является ли `X` словом из заданного списка, аналогично для предикатов `have(X)` и `question_word(X)`, предикат `have_lst([X], REL)`, проверяет ялвляется ли `X` членом списка состоящего из `REL`.

```prolog
word(X) :- member(X, [whose, "Whose"]).
have(X) :- member(X, [is, "Is"]).
have_lst([X], REL) :- member(X, [REL]).
question_word(X) :-  member(X, ['?']).

question(L) :- L = [IS, NAME_0, REL, NAME_1, Q], have(IS), relative(X, NAME_0, NAME_1),!, have_lst(X, REL),
    question_word(Q), write(NAME_0),  write(" is "),  write(REL),  write(" "),  write(NAME_1), write(".").

question(L) :- L = [WHOSE, REL, IS, NAME, Q], word(WHOSE), fam(REL, NAME, ANS), have(IS), question_word(Q),
    write(NAME), write(" is "),  write(REL), write(" "), write(ANS), write("."), nl.
```

Примеры использования:
```prolog
2 ?- question([is, 'Evgenii_Pishchik', son, 'Natalia_Samara', '?']). 
Evgenii_Pishchik is son Natalia_Samara.
true.

3 ?- question([is, 'Sergei_Pishchik', son, 'Natalia_Samara', '?']).  
false.
```

```prolog
4 ?- question([whose, father, is, 'Sergei_Pishchik', '?']).                         
Sergei_Pishchik is father Evgenii_Pishchik.
true ;
Sergei_Pishchik is father Aleksey_Pishchik.
true ;
false.

5 ?- question([whose, father, is, 'Evgenii_Pishchik', '?']).
false.
```

## Выводы

Prolog яляется логическим языков программирования, что дает свои плюсы и минусы, создание предикатов является логически понятным и интуитивно простым для человнка, т.к. по сути мы так размышляем, в прологе есть правила и факты, факт это что-то известное, то что мы знаем, правило - закономерность, базирующаяся на фактах, запрос пролога представляет обход дерева и бэктрекинг. В прологе есть механизм отсечения, который позволяет ускорять работу программы, убирать лишние вхождения, на прологе можно писать многие стандартные алгоритмы, такие как dfs, bfs, с его помощью удобно решать логические задачки, также Prolog удобно использовать для описания некоторых математических предикатов. Я научился в Prolog-е работать со списками, писать различные предикаты отношений, объединять предикаты в один общий предикат, использовать рекурсию, механизм отсечения. Prolog позволяет создавать программы с набором очень гибких предикатов, тем самым на нем достаточно просто писать простые программы, использующие в виде запроса естественный язык. 
