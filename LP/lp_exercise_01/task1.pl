%Пункт 3
my_length([],0).
my_length([_|T],N) :- my_length(T,K), N is K+1.

my_member(H, [H|_]). 
my_member(X, [_|T]) :- my_member(X, T).

my_append([], X, X).
my_append([H|T], X, [H|T_RES]) :- my_append(T, X, T_RES).

my_remove(X, [X|T], T).
my_remove(X, [H|T], [H|T1]) :- my_remove(X, T, T1).

my_permute([], []).
my_permute(L, [X|T]) :- my_remove(X, L, R), my_permute(R, T).

sub_start([], _) :- !.
sub_start([H|TSub], [H|TList]):- sub_start(TSub, TList).
my_sublist(Sub, List):- sub_start(Sub, List), !.
my_sublist(Sub, [_|T]):- my_sublist(Sub, T).

%Пункт 4 Вариант 2
%При помощи стандартных предикатов
delete_std(L, R) :- reverse(L, [_|T]), append([], T, F), reverse(F, R).
%При помощи своих предикатов
my_reverse(List, ReverseList) :- my_reverse(List, [], ReverseList).
my_reverse([], Buffer, Buffer) :- !.
my_reverse([Head|Tail], Buffer, ReverseList) :- my_reverse(Tail, [Head|Buffer], ReverseList).
my_delete(L, R) :- my_reverse(L, [_|T]), my_append([], T, F), my_reverse(F, R).

%Пункт 5 Вариант 6
even(X) :- X mod 2 < 1.
count_even_std([], C) :- C is 0.
count_even_std([H|T], C) :- even(H) -> count_even_std(T, N), C is N+1; count_even_std(T, N1), C is N1.

%Пункт 6
%Можно использовать удаление последнего элемента вместе с методом my_append, чтобы менять последний элемент
%Можно использовать удаление последнего элемента вместе с методом my_append и my_reverse, чтобы менять первый элемент
change_last(X, L, R) :- delete_std(L, F), my_append(F, [X], R).
change_first(X, L, R) :- my_reverse(L, F), delete_std(F, E), my_append(E, [X], T), my_reverse(T, R).