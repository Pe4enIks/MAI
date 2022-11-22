%для собрки всего в одну команду, кроме to_term_fam, перед таким запросом нужно запустить delete_symbols.py
step_1 :-
    parse_sex,
    parse_fam_fem,
    parse_fam_m,
    delete,
    get_child,
    get_id_name,
    to_terms_sex.
%после преобразования delete_symbols.py
step_2:-
    to_term_fam,
    %to_term_fam_not_working,
    res_union.
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
%тут пытался сделать через аналогию двойного цикла, но так и не получилось, работает только для одной строки, тоесть одного цикла
%второй цикл удален, чтобы работало для одной строки, иначе никогда не заканичивается
to_term_fam_not_working :-
    open('fam_pairs.ged', read, In_1),
    open('id_name.ged', read, In_2),
    open('res_fam.pl', write, Out),
    read_line_to_string(In_1, X),
    split_string(X, ",", ",", M_STR),
    change_my(M_STR, Out, In_2),
    close(In_1),
    close(In_2),
    close(Out).
change_my(M_STR_0, Out, In_0) :- 
    nth0(0, M_STR_0, CHILD), 
    nth0(1, M_STR_0, PARENT), 
    repeat, 
    read_line_to_string(In_0, Y),
    split_string(Y, ",", ",", M_STR_1),
    find_my(M_STR_1, Out, CHILD, PARENT),
    nl,
    Y=end_of_file, !,
    nl.
find_my(M_STR, Out, CHILD, PARENT) :- nth0(0, M_STR, ID), nth0(1, M_STR, NAME), CHILD=ID -> 
    write(Out, "child("), write(Out, NAME), write(Out, ","); writef("").
find_my(M_STR, Out, CHILD, PARENT) :- nth0(0, M_STR, ID), nth0(1, M_STR, NAME), PARENT=ID ->
   write(Out, NAME), write(Out, ").\n"); writef("").
%конец неработающей части
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