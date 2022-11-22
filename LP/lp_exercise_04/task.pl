% Пищик Е.С. М8О-206Б-19
% Вариант 10
name("Женя").
name("Андрей").
name("Олег").

subjects("бананы").
subjects("орехи").
subjects("яблоки").

opposition(X, Y) :- append(_, ["но"|Y], X).
opposition(X, Y) :- append(Y, ["но"|_], X).
opposition(X, Y) :- append(_, ["а"|Y], X).
opposition(X, Y) :- append(Y, ["а"|_], X).

like(X, Y) :- append(_, ["любит"|Y], X).
not_like(X, Y) :- append(_, ["не", "любит"|Y], X).

subject([H|_], H, _) :- subjects(H).
subject([_|T], X, Y) :- subject(T, X, Y).

like_name(X, likes(Y, Z), Y) :- subject(X, Z, Y).
not_like_name(X, not_likes(Y, Z), Y) :- subject(X, Z, Y).

my_phrase(X, Y, Z) :- not(not_like(Y,K)), like(Y, K), like_name(K, Z, X).
my_phrase(X, Y, Z) :- not_like(Y, K), not_like_name(K, Z, X).
opposition_phrase(X, Y, Z) :- opposition(Y, K), my_phrase(X, K, Z).

decompose([H|T], X) :- name(H), opposition_phrase(H, T, X).
decompose([H|T], X) :- name(H), not(opposition_phrase(H, T, X)), my_phrase(H, T, X).
