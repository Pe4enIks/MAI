% Пищик Е.С. М8О-206Б-19.
% Лабораторная работа №3.
% Вариант 5.
% Вдоль доски расположено 7 лунок, в которых лежат 3 черных и 3 белых шара.
% Передвинуть черные шары на место белых, а белые - на место черных.
% Шар можно передвинуть в соседнюю с ним пустую лунку, либо в пустую лунку, находящуюся непосредсвенно за ближайшим шаром.

% Предикат добавления элемента к списку.
append([], L, L).
append([H|T], L, [H|TRES]) :- append(T, L, TRES).

% Переходы между состояниями.
move(A,B) :- append(H, ['_','w'|T], A), append(H, ['w','_'|T], B).
move(A,B) :- append(H, ['w','_'|T], A), append(H, ['_','w'|T], B).
move(A,B) :- append(H, ['_','b'|T], A), append(H, ['b','_'|T], B).
move(A,B) :- append(H, ['b','_'|T], A), append(H, ['_','b'|T], B).
move(A,B) :- append(H, ['_','b','w'|T], A), append(H, ['w','b','_'|T], B).
move(A,B) :- append(H, ['_','w','b'|T], A), append(H, ['b','w','_'|T], B).
move(A,B) :- append(H, ['b','w','_'|T], A), append(H, ['_','w','b'|T], B).
move(A,B) :- append(H, ['w','b','_'|T], A), append(H, ['_','b','w'|T], B).

prolong([X|T], [Y,X|T]) :- move(X, Y), not(member(Y, [X|T])).

% Печать результата на экран.
print([_]).
print([H|T]) :- print(T), nl, write(H).

% Поиск в ширину.
bfs(X, Y) :- write(X), bdth([[X]], Y, Z), print(Z), !.
bdth([[H|T]|_], H, [H|T]).
bdth([H|T], X, Z) :- findall(W, prolong(H, W), Y), append(T, Y, E), !, bdth(E, X, Z).
bdth([_,T], X, Y) :- bdth(T, X, Y).

% Поиск в глубину.
dfs(X, Y) :- write(X), ddth([[X]], Y, Z), print(Z), !.
ddth([[H|T]|_], H, [H|T]).
ddth([H|T], X, Z) :- findall(W, prolong(H, W), Y), append(Y, T, E), !, ddth(E, X, Z).
ddth([_,T], X, Y) :- ddth(T, X, Y).

% Поиск с итерационным погружением.
search_id(X, Y, W, D) :- depth_id([X], Y, W, D).
depth_id([H|T], H, [H|T], 0).
depth_id(W, X, Y, N) :- N>0, prolong(W, NEWW), N1 is N-1, depth_id(NEWW, X, Y, N1).
search_id(X, Y, W) :- int(LEV), search_id(X, Y, W, LEV).
search_id(X, Y) :- write(X), search_id(X, Y, Z), print(Z), !.
int(1).
int(N) :- int(N1), N is N1+1.
