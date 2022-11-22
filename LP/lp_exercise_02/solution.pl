% ����� �.�. �8�-206�-19
% ������������ ������ �2
% ������� 20

% �����
name("����").
name("����").
name("����").
name("����").
name("����").

% ���������� ��� ������ �������
gender("����", f).
gender("����", f).
gender("����", m).
gender("����", m).
gender("����", m).

% ��������� ��������
same_gender(X, Y) :- gender(X, Z), gender(Y, Z).
% ��������� ������
diff_gender(X, Y) :- not(same_gender(X, Y)).

% ��������������� ��������
my_unique([]) :- !.
my_unique([H|T]) :- member(H, T), !, fail; my_unique(T).

% ���������� ����
without_mom("����").
% ������� ����
has_mom(X) :- not(without_mom(X)).

% �������� ���� ������� �� ����������� � ���������� ����
parents_not_knows("����", "����").
parents_knows(X, Y) :- not(parents_not_knows(X, Y)), not(parents_not_knows(Y, X)).

% ���� ���� ����������� � ���������� �������� -> ���� != ��������
not_name("����", "��������").
my_not(X, Y) :- not(not_name(X, Y)).

solve(RES) :-
	RES = [N1 = "��������",	N2 = "��������", N3 = "�������", N4 ="��������", N5 = "��������"],
	name(N1), name(N2),	name(N3), name(N4),	name(N5),
	% �������� �� ������������ ����
	my_unique([N1, N2, N3, N4, N5]),
	% �������� � �������� ������ � ����� ������������� ������� -> ��������� ���
	same_gender(N5, N1),
	% ���� �������� ������ � ������ �������� -> � �������� � �������� ���� ������ � ��� �������
	has_mom(N5),
	has_mom(N2),
	parents_knows(N5, N2),
	% ��� �������� - �������
	gender(N2, m),
	% ��� �������� != ����
	my_not(N2, "��������"),
	% �� ���������, ��� ���� ���� ��� ����������� � ���������� �������� -> ��� �������
	parents_knows(N2, "����"),
	% ���� � ���� ������� ������� ������ ��������� ��������. ��� ������� ����� ��������,
	% ��� �� ���� ���������� ���������� -> ������� � �������� ������� ����
	diff_gender(N3, N1),
	% ���� � ���� ������� ������� ������ ��������� ��������. ��� ������� ����� ��������,
	% ��� �� ���� ���������� ���������� -> � ������� � �������� ���� ������ � �� �������� �������
	has_mom(N3),
	has_mom(N1),
	parents_knows(N1, N3).