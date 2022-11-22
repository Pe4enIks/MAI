%факты полученные парсингом
male('Evgenii_Pishchik').
male('Sergei_Pishchik').
female('Natalia_Samara').
male('Aleksey_Pishchik').
female('Eva_Samara').
male('Petr_Samara').
male('Nikolai_Pishchik').
female('Nina_Pishchik').
female('Anna_Samara').
male('Aleksandr_Samara').
male('Sergei_Lyachovec').
male('Nikolai_Lyachovec').
female('Veronika_Lyachovec').
male('Iakov_Samara').
female('Varvara_Samara').
male('Adam_Los').
female('Maria_Los').
male('Nikon_Burim').
male('Lucyan_Drozd').
female('Tatyana_Pishchik').
male('Fedor_Pishchik').
male('Ivan_Kozlov').
female('Aleksandra_Kozlova').

child('Evgenii_Pishchik','Natalia_Samara').
child('Aleksey_Pishchik','Natalia_Samara').
child('Natalia_Samara','Eva_Samara').
child('Anna_Samara','Eva_Samara').
child('Aleksandr_Samara','Eva_Samara').
child('Sergei_Pishchik','Nina_Pishchik').
child('Sergei_Lyachovec','Anna_Samara').
child('Veronika_Lyachovec','Anna_Samara').
child('Petr_Samara','Varvara_Samara').
child('Eva_Samara','Maria_Los').
child('Nikolai_Pishchik','Tatyana_Pishchik').
child('Nina_Pishchik','Aleksandra_Kozlova').
child('Evgenii_Pishchik','Sergei_Pishchik').
child('Aleksey_Pishchik','Sergei_Pishchik').
child('Natalia_Samara','Petr_Samara').
child('Anna_Samara','Petr_Samara').
child('Aleksandr_Samara','Petr_Samara').
child('Sergei_Pishchik','Nikolai_Pishchik').
child('Sergei_Lyachovec','Nikolai_Lyachovec').
child('Veronika_Lyachovec','Nikolai_Lyachovec').
child('Petr_Samara','Iakov_Samara').
child('Eva_Samara','Adam_Los').
child('Varvara_Samara','Nikon_Burim').
child('Maria_Los','Lucyan_Drozd').
child('Nikolai_Pishchik','Fedor_Pishchik').
child('Nina_Pishchik','Ivan_Kozlov').

%пункт 3 КП, вариант 6 - тёща
father(X, Y) :- child(Y, X), male(X), X\=Y.
mother(X, Y) :- child(Y, X), female(X), X\=Y.
husband(X, Y) :- child(Z, X), child(Z, Y), male(X), X\=Y.
wife(X, Y) :- child(Z, X), child(Z, Y), female(X), X\=Y.

%является ли X тёщей Y
mother_in_law(X, Y) :- mother(X, Z), wife(Z, Y), male(Y), X\=Y.

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


perm(X, Y) :- fam(_, X, Y).
prlng([X|T], [Y, X|T]) :- perm(X, Y), not(member(Y, [X|T])).

bfs([[H|T]|_], H, [H|T]).
bfs([H|T], RES, TREE) :- findall(W, prlng(H, W), TREES), append(T, TREES, NEWTREES), !, bfs(NEWTREES, RES, TREE).

find([_], R, R).
find([X,Y|T], R, REL) :- fam(RE, X, Y), find([Y|T], [RE|R], REL).

relative(REL, X, Y) :- bfs([[X]], Y, R), reverse(R, RE), find(RE, [], NEWREL), reverse(NEWREL, REL), N is 0.

%пункт 5 КП
%естественно-языковой интерфейс
word(X) :- member(X, [whose, "Whose"]).
have(X) :- member(X, [is, "Is"]).
have_lst([X], REL) :- member(X, [REL]).
question_word(X) :-  member(X, ['?']).
prev_set(NAME) :- nb_setval(name, NAME).
word_prev(X) :- member(X,["His",his,"Him",him,"Her",her,"She",she,"He",he]),!.
to_word(X) :- member(X, ["to", to]).


%Пример запроса: [is, name_0, relationship, name_1, ?]
question(L) :- L = [IS, NAME_0, REL, NAME_1, Q], have(IS), relative(X, NAME_0, NAME_1),!, have_lst(X, REL),
    question_word(Q), write(NAME_0),  write(" is "),  write(REL),  write(" "),  write(NAME_1), write(".").

%Пример запроса: [whose, relationship, is, name_1, ?]
question(L) :- L = [WHOSE, REL, IS, NAME, Q], word(WHOSE), fam(REL, NAME, ANS), have(IS), question_word(Q),
    write(NAME), write(" is "),  write(REL), write(" "), write(ANS), write("."), prev_set(NAME), nl.

%Пример запроса [she/he/her/him/his, relationship, to, name, ?] после пред запроса [whose, relationship, is, name_1, ?]
question(L) :- L = [NAME_0, REL, TO, NAME_1, Q], word_prev(NAME_0), nb_getval(name, NAME_2), relative(X, NAME_2, NAME_1),!,have_lst(X, REL),
to_word(TO), question_word(Q), write(NAME_2),  write(" is "),  write(REL),  write(" "),  write(NAME_1), write(".").
