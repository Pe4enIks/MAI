# Отчет по лабораторной работе №1
## Работа со списками и реляционным представлением данных
## по курсу "Логическое программирование"

### студент: Пищик Е.С.

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |               |
| Левинская М.А.|   26.10      |     5--       |

> *Комментарии проверяющих (обратите внимание, что более подробные комментарии возможны непосредственно в репозитории по тексту программы)*
В реализации предиката без использования стандартных предикатов в Вашем  предикате не должны присутствовать другие предикаты. Это должна быть рекурсивная реализация (обычно в 2 строки: факт + правило).

## Введение

В прологе список - структура определенная рекурсивно, у списка есть первый элемент - голова и элемент - хвост, являющийся также списком, списки в прологе напоминают деревья в императивных ЯП.

## Реализация стандартных предикатов
```prolog
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
```

## Задание 1.1: Предикат обработки списка

`my_delete(L,R)` - удаляет последний элемент списка, реализация без использования стандартных предикатов.

Примеры использования:
```prolog
?- my_delete([1,2,3,4], R).
R = [1, 2, 3].
```

Реализация:
```prolog
my_append([], X, X).
my_append([H|T], X, [H|T_RES]) :- my_append(T, X, T_RES).
my_reverse(List, ReverseList) :- my_reverse(List, [], ReverseList).
my_reverse([], Buffer, Buffer) :- !.
my_reverse([Head|Tail], Buffer, ReverseList) :- my_reverse(Tail, [Head|Buffer], ReverseList).
my_delete(L, R) :- my_reverse(L, [H|T]), my_append([], T, F), my_reverse(F, R).
```

Делаем реверс списка, к пустому списку добавляем хвост этого реверсного списка, и делаем реверс списка в результирующий список.

`delete_std(L, R)` - удаляет последний элемент списка, реализация с помощью стандартных предикатов.

Примеры использования:
```prolog
?- delete_std([1,2,3,4], R).
R = [1, 2, 3].
```

Реализация:
```prolog
delete_std(L, R) :- reverse(L, [H|T]), append([], T, F), reverse(F, R).
```

Делаем реверс списка, к пустому списку добавляем хвост этого реверсного списка, и делаем реверс списка в результирующий список.

## Задание 1.2: Предикат обработки числового списка

`count_even_std(L, C)` - подсчитывает число чётных чисел в списке L и заносит в переменную С.

Примеры использования:
```prolog
?- count_even_std([1,2,3,4,5,6,7,8,9], C).
C = 4.
```

Реализация:
```prolog
even(X) :- X mod 2 < 1.
count_even_std([], C) :- C is 0.
count_even_std([H|T], C) :- even(H) -> count_even_std(T, N), C is N+1; count_even_std(T, N1), C is N1.
```

Предикат even(X) - возвращает true, если число чётное и false иначе, count_even_std определяется рекурсивно, если на вход подается пустой список, то C инициализируется 0, если список не пустой, для головы списка идёт проверка не чётность, если число чётное, то вызывается count_even_std(T, N), где T хвост списка, а N - счётчик, делаем C = N+1, аналогично если число нечетное, только C = N1, то есть C не увеличивается, если число нечётное (дополнить условием нечетности) и увеличивается, если чётное.

## Задание 2: Реляционное представление данных

Таблицы позволяют уменьшить нагрузку на пользователя при чтении таблицы, также зная что вам нужно изменить, вы меняете 1 таблицу с конкретными данными, а не мучаетесь с огромным количеством разнотиповых данных, таблицы - некоторая группировка данных, из минусов - не всегда удобно совмещать данные из таблиц, можно запутаться что и как соединять.

Первый предикат `res_in_group(X, L, R)`, при помощи bagof, получаем для каждого X - номера группы список L - имён, R вычисляем при помощи предиката `all_marks([H|T], RES, N))`, который рекурсивно считает сумму и количество оценок для списка людей, внутри себя он использует предикат `avg_mark([H|_], ANS)`, который считает средную оценку для каждого человека.

Примеры использования:
```prolog
?- res_in_group(Group,Table ,Avg_Mark). 
Group = 101,
Table = ['Петровский', 'Сидоров', 'Мышин', 'Безумников', 'Густобуквенникова'],
Avg_Mark = 3.9 ;
Group = 102,
Table = ['Петров', 'Ивановский', 'Биткоинов', 'Шарпин', 'Эксель', 'Текстописов', 'Криптовалютников', 'Азурин', 'Круглотличников'],
Avg_Mark = 3.7777777777777777 ;
Group = 103,
Table = ['Сидоркин', 'Эфиркина', 'Сиплюсплюсов', 'Программиро', 'Клавиатурникова', 'Решетников', 'Текстописова', 'Вебсервисов'],
Avg_Mark = 3.7708333333333335 ;
Group = 104,
Table = ['Иванов', 'Запорожцев', 'Джаво', 'Фулл', 'Круглосчиталкин', 'Блокчейнис'],
Avg_Mark = 3.861111111111111.
```

Реализация:
```prolog
sum_list([], 0, C) :- C is 0.
sum_list([H|T], Sum, N) :- sum_list(T, Rest, C), Sum is H + Rest, N is C+1.
avg_mark([H|_], ANS) :- findall(Z, grade(H, _, Z), R), sum_list(R, Sum, N), ANS is Sum/N.
all_marks([], RES, N) :- RES is 0, N is 0. 
all_marks([H|T], RES, N) :- avg_mark([H], ANS), all_marks(T, Y, K), N is K+1, RES is Y+ANS.
res_in_group(X, L, R) :- bagof(Y, student(X, Y), L), all_marks(L, RES, N), R is RES/N.
```

Второй предикат `bad_mark(S, N)` просто для каждого предмета выводит людей получивших оценку 2, при помощи bagof.

Примеры использования:
```prolog
?- bad_mark(Subject, Names).
Subject = 'ENG',
Names = ['Эфиркина'] ;
Subject = 'FP',
Names = ['Криптовалютников'] ;
Subject = 'INF',
Names = ['Эфиркина', 'Джаво', 'Безумников'] ;
Subject = 'LP',
Names = ['Запорожцев', 'Эфиркина', 'Текстописов'] ;
Subject = 'MTH',
Names = ['Запорожцев', 'Круглосчиталкин', 'Густобуквенникова', 'Криптовалютников', 'Блокчейнис', 'Азурин'] ;
Subject = 'PSY',
Names = ['Биткоинов', 'Текстописова', 'Криптовалютников', 'Азурин', 'Вебсервисов'].
```

Реализация:
```prolog
bad_mark(S, N) :- bagof(Z, grade(Z, S, 2), N).
```

Третий предикат `count_mark_2(X, C)` для каждой группы выводит количество людей получивших оценки 2, список людей для каждой группы определяется при помощи bagof, потом вызывается предикат `count_mark_2_in_group([H|T], C)`, который рекурсивно для списка людей считает количество людей получивших оценку 2.

Примеры использования:
```prolog
?- count_mark_2(Group, Count). 
Group = 101,
Count = 2 ;
Group = 102,
Count = 4 ;
Group = 103,
Count = 3 ;
Group = 104,
Count = 4.
```

Реализация:
```prolog
count_mark_2_in_group([], C) :- C is 0.
count_mark_2_in_group([H|T], C) :- grade(H, _, 2) -> count_mark_2_in_group(T, N), C is N+1; count_mark_2_in_group(T, N), C is N.
count_mark_2(X, C) :- bagof(Y, student(X, Y), L), count_mark_2_in_group(L, C).
```

## Выводы

Лабораторная работа научила меня работать со списками в SWI Prlog и реляционным представлением данных, логическое программирование в целом заставляет человека взглянуть на программирование с другой стороны, начать думать по иному, что безусловно полезно, я задумался над тем, что мне кажется Prolog не совсем всё-таки подходит для работы с реляционными данными, SQL и т.п. всё-таки будут удобнее. После этой лабораторной работы стало более понятно как работает рекурсия в Prolog-е и как пролог осуществляет запросы. Также стало намного понятней, как устроена арифметика в Prlog-е и как работать с ней в рекурсивных вызовах. 