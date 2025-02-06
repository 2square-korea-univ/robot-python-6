---
title: "내장 함수 및 표준 라이브러리"
css: style.css
tags:
  - datetime
---

## 1) 문자열 메소드

```python
text = "Hello, World!"

# 문자열 길이 반환
print(len(text)) # 13

# 소문자로 변환
print(text.lower()) # "hello, world!"

# 대문자로 변환
print(text.upper()) # "HELLO, WORLD!"

# 특정 문자열 포함 여부 확인
print("World" in text) # True

# 문자열 치환
print(text.replace("World", "Python")) # "Hello, Python!"
```

---

## 2) 리스트 메소드

리스트 자료형(list)에도 유용한 메소드들이 있습니다.

```python
my_list = [1, 2, 3, 4, 5]

# 리스트에 요소 추가
my_list.append(6)
print(my_list) # [1, 2, 3, 4, 5, 6]

# 리스트에서 요소 제거
my_list.remove(3)
print(my_list) # [1, 2, 4, 5, 6]

# 리스트 정렬
my_list.sort()
print(my_list) # [1, 2, 4, 5, 6]

# 리스트 뒤집기
my_list.reverse()
print(my_list) # [6, 5, 4, 2, 1]
```

---

## 3) 딕셔너리 메소드

```python
my_dict = {"name": "Alice", "age": 25, "city": "New York"}

# 키-값 쌍 추가
my_dict["email"] = "alice@example.com"
print(my_dict) # {'name': 'Alice', 'age': 25, 'city': 'New York', 'email': 'alice@example.com'}

# 특정 키의 값 가져오기
print(my_dict.get("name")) # "Alice"

# 모든 키 가져오기
print(my_dict.keys()) # dict_keys(['name', 'age', 'city', 'email'])

# 모든 값 가져오기
print(my_dict.values()) # dict_values(['Alice', 25, 'New York', 'alice@example.com'])
```

---

## 4) 집합 메소드

```python
my_set = {1, 2, 3, 4, 5}

# 요소 추가
my_set.add(6)
print(my_set) # {1, 2, 3, 4, 5, 6}

# 요소 제거
my_set.remove(3)
print(my_set) # {1, 2, 4, 5, 6}

# 다른 집합과의 합집합
another_set = {4, 5, 6, 7, 8}
print(my_set.union(another_set)) # {1, 2, 4, 5, 6, 7, 8}

# 다른 집합과의 교집합
print(my_set.intersection(another_set)) # {4, 5, 6}
```

---

## 자료형 별 메소드 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 자료형 | 메소드                          | 설명                                                    | 실행 예시                                                     |
| ------ | ------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
| 리스트 | `append(x)`                     | 리스트 끝에 요소 x 추가                                 | `lst = [1, 2]; lst.append(3); print(lst) # [1, 2, 3]`         |
| 리스트 | `extend(iterable)`              | 리스트 끝에 다른 iterable의 모든 요소 추가              | `lst = [1, 2]; lst.extend([3, 4]); print(lst) # [1, 2, 3, 4]` |
| 리스트 | `insert(i, x)`                  | 지정한 위치 i에 요소 x 삽입                             | `lst = [1, 3]; lst.insert(1, 2); print(lst) # [1, 2, 3]`      |
| 리스트 | `remove(x)`                     | 리스트에서 첫 번째로 나오는 요소 x 제거                 | `lst = [1, 2, 3, 2]; lst.remove(2); print(lst) # [1, 3, 2]`   |
| 리스트 | `pop([i])`                      | 지정한 위치 i의 요소 제거 및 반환 (기본값: 마지막 요소) | `lst = [1, 2, 3]; print(lst.pop()) # 3; print(lst) # [1, 2]`  |
| 리스트 | `clear()`                       | 리스트의 모든 요소 제거                                 | `lst = [1, 2, 3]; lst.clear(); print(lst) # []`               |
| 리스트 | `index(x[, start[, end]])`      | 리스트에서 첫 번째로 나오는 요소 x의 인덱스 반환        | `lst = [1, 2, 3]; print(lst.index(2)) # 1`                    |
| 리스트 | `count(x)`                      | 리스트에서 요소 x의 개수 반환                           | `lst = [1, 2, 2, 3]; print(lst.count(2)) # 2`                 |
| 리스트 | `sort(key=None, reverse=False)` | 리스트를 정렬                                           | `lst = [3, 1, 2]; lst.sort(); print(lst) # [1, 2, 3]`         |
| 리스트 | `reverse()`                     | 리스트의 요소 순서 반전                                 | `lst = [1, 2, 3]; lst.reverse(); print(lst) # [3, 2, 1]`      |

</div>

---

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 자료형   | 메소드                | 설명                                       | 실행 예시                                                                               |
| -------- | --------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------- |
| 딕셔너리 | `keys()`              | 딕셔너리의 모든 키 반환                    | `my_dict = {'a': 1, 'b': 2}; print(my_dict.keys()) # dict_keys(['a', 'b'])`             |
| 딕셔너리 | `values()`            | 딕셔너리의 모든 값 반환                    | `my_dict = {'a': 1, 'b': 2}; print(my_dict.values()) # dict_values([1, 2])`             |
| 딕셔너리 | `items()`             | 딕셔너리의 모든 키-값 쌍 반환              | `my_dict = {'a': 1, 'b': 2}; print(my_dict.items()) # dict_items([('a', 1), ('b', 2)])` |
| 딕셔너리 | `get(key[, default])` | 지정한 키의 값 반환 (기본값: None)         | `my_dict = {'a': 1}; print(my_dict.get('a')) # 1; print(my_dict.get('b', 0)) # 0`       |
| 딕셔너리 | `pop(key[, default])` | 지정한 키의 값 반환 및 해당 키-값 쌍 제거  | `my_dict = {'a': 1, 'b': 2}; print(my_dict.pop('a')) # 1; print(my_dict) # {'b': 2}`    |
| 딕셔너리 | `update([other])`     | 다른 딕셔너리나 키-값 쌍으로 딕셔너리 갱신 | `my_dict = {'a': 1}; my_dict.update({'b': 2}); print(my_dict) # {'a': 1, 'b': 2}`       |

</div>

</div>

---

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 자료형 | 메소드                        | 설명                                        | 실행 예시                                                                     |
| ------ | ----------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------- |
| 집합   | `add(x)`                      | 집합에 요소 x 추가                          | `s = {1, 2}; s.add(3); print(s) # {1, 2, 3}`                                  |
| 집합   | `remove(x)`                   | 집합에서 요소 x 제거 (없으면 KeyError 발생) | `s = {1, 2, 3}; s.remove(2); print(s) # {1, 3}`                               |
| 집합   | `discard(x)`                  | 집합에서 요소 x 제거 (없으면 무시)          | `s = {1, 2, 3}; s.discard(4); print(s) # {1, 2, 3}`                           |
| 집합   | `pop()`                       | 집합에서 임의의 요소 제거 및 반환           | `s = {1, 2, 3}; print(s.pop()) # 1 (또는 2 또는 3); print(s) # {2, 3}`        |
| 집합   | `clear()`                     | 집합의 모든 요소 제거                       | `s = {1, 2, 3}; s.clear(); print(s) # set()`                                  |
| 집합   | `union(*others)`              | 여러 집합의 합집합 반환                     | `s1 = {1, 2}; s2 = {2, 3}; print(s1.union(s2)) # {1, 2, 3}`                   |
| 집합   | `intersection(*others)`       | 여러 집합의 교집합 반환                     | `s1 = {1, 2, 3}; s2 = {2, 3, 4}; print(s1.intersection(s2)) # {2, 3}`         |
| 집합   | `difference(*others)`         | 여러 집합의 차집합 반환                     | `s1 = {1, 2, 3}; s2 = {2, 3, 4}; print(s1.difference(s2)) # {1}`              |
| 집합   | `symmetric_difference(other)` | 두 집합의 대칭 차집합 반환                  | `s1 = {1, 2, 3}; s2 = {2, 3, 4}; print(s1.symmetric_difference(s2)) # {1, 4}` |

</div>

---

# 39: 내장 함수 (Built-in Functions) 개요

파이썬에는 별도의 import 없이 바로 사용할 수 있는 내장 함수들이 제공됩니다.
이들은 다양한 기능을 수행하며, 파이썬 프로그래밍 효율을 높여줍니다.

<br>
예시

- `len()`: 자료형(문자열, 리스트 등)의 길이를 반환
- `range()`: 특정 범위의 정수 시퀀스를 만들어줌
- `type()`: 객체(변수)의 자료형을 알려줌
- `print()`: 콘솔에 출력
- `input()`: 콘솔에서 사용자 입력 받음
- `sum()`, `min()`, `max()`
- `abs()`, `round()`
- `enumerate()`
- `map()`, `filter()`, `zip()`
- (이 외에도 매우 많습니다!)

---

# 40: 자주 사용하는 내장 함수 살펴보기

## 1) `len()`, `type()`

```python
my_list = [1, 2, 3]
print(len(my_list)) # 3
print(type(my_list)) # <class 'list'>
```

---

## 2) `sum()`, `min()`, `max()`

```python
numbers = [5, 3, 8, 1]
print(sum(numbers)) # 17
print(min(numbers)) # 1
print(max(numbers)) # 8
```

---

## 3) `abs()`, `round()`

```python
x = -3.5
print(abs(x)) # 3.5
print(round(x)) # -4
print(round(3.14159, 2)) # 3.14
```

---

# 41: 고급 내장 함수 - `enumerate()`, `map()`, `filter()`

## 1) `enumerate()`

리스트(또는 다른 반복 가능한 객체)를 순회하면서 인덱스(index)와 값을 함께 반환

```python
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits):
  print(idx, fruit)
```

---

## 2) `map()`

반복 가능한 객체의 각 요소에 함수를 적용하여 새로운 반복 객체를 생성

```python
numbers = [1, 2, 3, 4]

# 모든 원소에 2배

# 모든 원소에 2배
iterable_object = [1, 2, 3, 4]
# map(함수, iterable_object)
doubled = list(map(int, iterable_object))
print(doubled) # [1, 2, 3, 4]
```

---

## 3) `filter()`

반복 가능한 객체에서 특정 조건을 만족하는 요소만 걸러내는 기능

```python
numbers = [1, 2, 3, 4, 5, 6]

# 짝수만 필터링

evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens)) # [2, 4, 6]
```

---

# 42: lambda 함수

익명 함수(anonymous function)라고도 부르며, 함수 이름이 없는 함수입니다.
비교적 간단하며 재사용되지 않는 함수에 사용합니다.

- 특히 메모리 관리에 효율적이므로 언어 기초에서는 주로 다루지 않습니다.

```python
def add_one(x):
  return x + 1
print(add_one(1)) # 2

add_one = lambda x: x + 1
print(add_one(1)) # 2
```

```python
# 함수를 지정해도 되지만, 이전에 생성하여야 함
map(add_one, [1, 2, 3])

# 람다 함수로 바로 작성해도 됨
map(lambda x: x + 1, [1, 2, 3])
```

---

# 예제 실습 (내장 함수)

[실습1]

문자열 리스트가 주어졌을 때, 각 문자열의 길이를 구한 뒤 그 합을 출력해 보세요.

예: ["apple", "banana", "kiwi"] -> 길이 합: 5 + 6 + 4 = 15

힌트: `len()`, `sum()`, `map()` 등을 활용

사용자로부터 여러 개의 수를 입력받아, 그 중 짝수만 걸러내어 리스트로 저장하고 출력해 보세요.

힌트: `input()`, `split()`, `map()`, `filter()` 이용

```python

# 실습 예시 코드 (힌트)

# words = ["apple", "banana", "kiwi"]

# lengths = list(map(len, words))

# print(lengths) # [5, 6, 4]

# print(sum(lengths)) # 15

```

---

# 43: 파이썬 표준 라이브러리 개요

내장 함수와 더불어, 파이썬 표준 라이브러리(standard library) 역시 매우 중요합니다.

설치 없이 바로 import 해서 사용할 수 있는 라이브러리
데이터 처리, 날짜/시간, 운영체제 인터페이스, 네트워크 관련 등 다양한 기능을 제공

예시

- `math`, `random`, `datetime`, `time`
- `os`, `sys`, `shutil`
- `re`, `json`, `csv`
- `itertools`, `functools`, `collections` 등

---

# 44: 수학 및 시간 라이브러리

## 1) `math` 모듈

```python
import math

print(math.sqrt(16)) # 4.0
print(math.pi) # 3.141592653589793
print(math.factorial(5)) # 120
```

---

## math 모듈 스크롤러블 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 함수                | 설명                          | 예시 코드                         |
| ------------------- | ----------------------------- | --------------------------------- |
| `math.sqrt(x)`      | x의 제곱근을 반환             | `math.sqrt(16)` -> 4.0            |
| `math.pow(x, y)`    | x의 y 제곱을 반환             | `math.pow(2, 3)` -> 8.0           |
| `math.factorial(x)` | x의 팩토리얼을 반환           | `math.factorial(5)` -> 120        |
| `math.gcd(x, y)`    | x와 y의 최대공약수를 반환     | `math.gcd(48, 18)` -> 6           |
| `math.sin(x)`       | x(라디안)의 사인 값을 반환    | `math.sin(math.pi / 2)` -> 1.0    |
| `math.cos(x)`       | x(라디안)의 코사인 값을 반환  | `math.cos(math.pi)` -> -1.0       |
| `math.tan(x)`       | x(라디안)의 탄젠트 값을 반환  | `math.tan(math.pi / 4)` -> 1.0    |
| `math.log(x)`       | x의 자연 로그(밑이 e)를 반환  | `math.log(math.e)` -> 1.0         |
| `math.log10(x)`     | x의 상용 로그(밑이 10)를 반환 | `math.log10(100)` -> 2.0          |
| `math.exp(x)`       | e의 x 제곱을 반환             | `math.exp(2)` -> 7.38905609893065 |
| `math.degrees(x)`   | 라디안을 도 단위로 변환       | `math.degrees(math.pi)` -> 180.0  |
| `math.radians(x)`   | 도를 라디안 단위로 변환       | `math.radians(180)` -> math.pi    |

</div>

##

---

## 2) `random` 모듈

```python
import random

print(random.random()) # 0.0 이상 1.0 미만의 임의의 실수
print(random.randint(1, 10)) # 1 이상 10 이하의 임의의 정수
print(random.choice(['가위','바위','보'])) # 리스트 중 임의 선택
```

---

## random 모듈 스크롤러블 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 함수                    | 설명                                   | 예시 코드                                  |
| ----------------------- | -------------------------------------- | ------------------------------------------ |
| `random.randint(a, b)`  | a와 b 사이의 정수를 반환               | `random.randint(1, 10)` -> 7               |
| `random.random()`       | 0과 1 사이의 임의의 부동 소수점을 반환 | `random.random()` -> 0.374                 |
| `random.uniform(a, b)`  | a와 b 사이의 임의의 부동 소수점을 반환 | `random.uniform(1.0, 10.0)` -> 5.5         |
| `random.choice(seq)`    | 시퀀스에서 임의의 요소를 반환          | `random.choice(['a', 'b', 'c'])` -> 'b'    |
| `random.shuffle(seq)`   | 시퀀스의 요소들을 임의로 섞음          | `random.shuffle([1, 2, 3])` -> [3, 1, 2]   |
| `random.sample(seq, k)` | 시퀀스에서 k개의 임의의 요소를 반환    | `random.sample(range(10), 3)` -> [2, 5, 8] |
| `random.seed(a)`        | 난수 생성기의 시드 설정                | `random.seed(10)`                          |

</div>

---

## 3) `datetime` 모듈

```python
import datetime

now = datetime.datetime.now()
print(now) # 현재 날짜와 시간
```

---

# datetime

## datetime 모듈 스크롤러블 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 함수                                    | 설명                                      | 예시 코드                                                                          |
| --------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `datetime.datetime.now()`               | 현재 날짜와 시간을 반환                   | `datetime.datetime.now()` -> 2023-03-15 12:34:56                                   |
| `datetime.datetime.strptime()`          | 문자열을 datetime 객체로 변환             | `datetime.datetime.strptime('2023-03-15', '%Y-%m-%d')`                             |
| `datetime.datetime.strftime()`          | datetime 객체를 문자열로 변환             | `datetime.datetime.now().strftime('%Y-%m-%d')` -> '2023-03-15'                     |
| `datetime.timedelta(days=1)`            | 시간 차이를 나타내는 객체 생성            | `datetime.datetime.now() + datetime.timedelta(days=1)`                             |
| `datetime.date.today()`                 | 현재 날짜를 반환                          | `datetime.date.today()` -> 2023-03-15                                              |
| `datetime.date(year, month, day)`       | 특정 날짜를 나타내는 date 객체 생성       | `datetime.date(2023, 3, 15)`                                                       |
| `datetime.time(hour, minute, second)`   | 특정 시간을 나타내는 time 객체 생성       | `datetime.time(12, 34, 56)`                                                        |
| `datetime.datetime.combine(date, time)` | date와 time을 결합하여 datetime 객체 생성 | `datetime.datetime.combine(datetime.date(2023, 3, 15), datetime.time(12, 34, 56))` |

</div>

---

## 4) `time` 모듈

```python
import time

time.sleep(2) # 2초 간 대기
print("2초 후 출력")
```

---

## time 모듈 스크롤러블 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 함수                            | 설명                                   | 예시 코드                                                     |
| ------------------------------- | -------------------------------------- | ------------------------------------------------------------- |
| `time.time()`                   | 현재 시간을 초 단위로 반환             | `time.time()` -> 1678901234.56789                             |
| `time.sleep(secs)`              | 주어진 초만큼 일시 중지                | `time.sleep(2)`                                               |
| `time.ctime([secs])`            | 초 단위 시간을 문자열로 변환           | `time.ctime(1678901234)` -> 'Wed Mar 15 12:34:56 2023'        |
| `time.strftime(format[, t])`    | 시간 객체를 문자열로 변환              | `time.strftime('%Y-%m-%d', time.localtime())` -> '2023-03-15' |
| `time.strptime(string, format)` | 문자열을 시간 객체로 변환              | `time.strptime('2023-03-15', '%Y-%m-%d')`                     |
| `time.localtime([secs])`        | 초 단위 시간을 로컬 시간으로 변환      | `time.localtime(1678901234)`                                  |
| `time.gmtime([secs])`           | 초 단위 시간을 UTC 시간으로 변환       | `time.gmtime(1678901234)`                                     |
| `time.mktime(t)`                | 로컬 시간 객체를 초 단위 시간으로 변환 | `time.mktime(time.localtime())`                               |

</div>

---

# 45: 운영체제 관련 라이브러리

## 1) `os` 모듈

운영체제 기능을 파이썬에서 사용 가능 (폴더/파일 접근, 환경 변수 등)

```python
import os

print(os.getcwd()) # 현재 작업 디렉토리 확인
print(os.listdir(".")) # 현재 디렉토리 목록
os.mkdir("test_folder") # 새 폴더 생성
```

---

## os 모듈 스크롤러블 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 함수                         | 설명                                    | 예시 코드                               |
| ---------------------------- | --------------------------------------- | --------------------------------------- |
| `os.getcwd()`                | 현재 작업 디렉토리 반환                 | `os.getcwd()` -> '/home/user'           |
| `os.listdir(path)`           | 지정된 디렉토리의 파일 및 디렉토리 목록 | `os.listdir('.')` -> ['file1', 'file2'] |
| `os.mkdir(path)`             | 새 디렉토리 생성                        | `os.mkdir('new_folder')`                |
| `os.remove(path)`            | 파일 삭제                               | `os.remove('file.txt')`                 |
| `os.rmdir(path)`             | 빈 디렉토리 삭제                        | `os.rmdir('empty_folder')`              |
| `os.rename(src, dst)`        | 파일 또는 디렉토리 이름 변경            | `os.rename('old_name', 'new_name')`     |
| `os.path.exists(path)`       | 경로의 존재 여부 확인                   | `os.path.exists('file.txt')` -> True    |
| `os.path.join(path, *paths)` | 경로 병합                               | `os.path.join('folder', 'file.txt')`    |

</div>

---

## 2) `sys` 모듈

인터프리터 관련 정보, 시스템과 관련된 정보를 다룰 수 있음

```python
import sys

print(sys.version) # 파이썬 버전
print(sys.platform) # OS 플랫폼 정보
```

---

## sys 모듈 스크롤러블 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 함수              | 설명                               | 예시 코드                           |
| ----------------- | ---------------------------------- | ----------------------------------- |
| `sys.version`     | 현재 파이썬 인터프리터의 버전 정보 | `sys.version` -> '3.8.5'            |
| `sys.platform`    | 현재 운영체제 플랫폼 정보          | `sys.platform` -> 'win32'           |
| `sys.exit([arg])` | 프로그램 종료                      | `sys.exit(0)`                       |
| `sys.argv`        | 명령줄 인수 리스트                 | `sys.argv` -> ['script.py', 'arg1'] |
| `sys.path`        | 모듈 검색 경로 리스트              | `sys.path` -> ['path1', 'path2']    |
| `sys.stdin`       | 표준 입력                          | `sys.stdin.read()`                  |
| `sys.stdout`      | 표준 출력                          | `sys.stdout.write('hello')`         |
| `sys.stderr`      | 표준 에러 출력                     | `sys.stderr.write('error')`         |

</div>

---

## 3) `shutil` 모듈

파일/디렉토리 복사, 이동 등 고수준의 파일 작업 가능

```python
import shutil

# 폴더 전체를 복사하는 예 (조심스럽게 사용)

shutil.copytree("test_folder", "test_folder_copy")
```

---

## shutil 모듈 스크롤러블 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 함수                        | 설명                                               | 예시 코드                                       |
| --------------------------- | -------------------------------------------------- | ----------------------------------------------- |
| `shutil.copy(src, dst)`     | 파일을 src에서 dst로 복사                          | `shutil.copy('file.txt', 'file_copy.txt')`      |
| `shutil.copy2(src, dst)`    | 메타데이터를 포함하여 파일을 src에서 dst로 복사    | `shutil.copy2('file.txt', 'file_copy.txt')`     |
| `shutil.copytree(src, dst)` | 디렉토리와 그 내용을 src에서 dst로 재귀적으로 복사 | `shutil.copytree('dir', 'dir_copy')`            |
| `shutil.move(src, dst)`     | 파일 또는 디렉토리를 src에서 dst로 이동            | `shutil.move('file.txt', 'new_location/')`      |
| `shutil.rmtree(path)`       | 디렉토리와 그 내용을 재귀적으로 삭제               | `shutil.rmtree('dir_to_delete')`                |
| `shutil.disk_usage(path)`   | 디스크 사용량 정보를 반환                          | `shutil.disk_usage('/')` -> (total, used, free) |
| `shutil.which(cmd)`         | 명령어의 경로를 반환                               | `shutil.which('python')` -> '/usr/bin/python'   |

</div>

---

# 46: 문자열, 파일 포맷 관련 라이브러리

## 1) `re` (정규 표현식)

문자열에서 특정 패턴을 찾거나, 치환할 때 강력한 기능 제공

```python
import re

pattern = r"\d+"
text = "My phone number is 010-1234-5678"
match = re.findall(pattern, text)
print(match) # ['010', '1234', '5678']
```

---

예를 들어, 전화번호를 찾아야 한다고 생각해봅시다.

- 010-1234-5678
- 010-123-4567
  이런 전화번호들을 찾으려면 어떻게 해야 할까요?

정규표현식을 사용하면:

```python
pattern = r"\d{3}-\d{3,4}-\d{4}"
```

이런 식으로 간단하게 표현할 수 있습니다.

여기서:

- \d는 숫자를 의미합니다
- {3}은 정확히 3번 반복
- {3,4}는 3번 또는 4번 반복
- -는 하이픈 문자 자체를 의미합니다

---

## 자주 사용되는 패턴

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 패턴 | 설명                          |
| ---- | ----------------------------- |
| \d   | 숫자 매칭 (0-9)               |
| \w   | 문자 매칭 (a-z, A-Z, 0-9, \_) |
| \s   | 공백 문자 매칭                |
| .    | 아무 문자나 매칭              |
| \*   | 0번 이상 반복                 |
| +    | 1번 이상 반복                 |
| ?    | 0번 또는 1번                  |

</div>

---

## 자주 사용되는 패턴 응용

정규표현식을 사용하여 다양한 패턴을 매칭할 수 있습니다. 몇 가지 예시를 살펴보겠습니다.

1. 이메일 주소 매칭

```python
email_pattern = r"[a-zA-Z0-9.%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

# example

print(re.match(email_pattern, "test@example.com")) # <re.Match object; span=(0, 15), match='test@example.com'>
print(re.match(email_pattern, "invalid-email@.com")) # None
```

2. 한국 전화번호 매칭

```python
phone_pattern = r"01[016789]-?\d{3,4}-?\d{4}"
```

3. URL 매칭

```python
url_pattern = r"https?://(?:www\.)?[\w-]+\.[\w.]+/?[\w-.~:/?#\[\]@!$&'()+,;=]"
```

---

정규표현식(Regular Expression) 상세 가이드

<div class="overflow-y-auto max-h-[400px] border rounded p-4">
1. 기본 패턴

```python
기본 문자 매칭
pattern1 = r"hello" # 'hello' 문자열과 정확히 일치
pattern2 = r"." # 모든 단일 문자와 일치 (줄바꿈 제외)
pattern3 = r"\." # 실제 마침표와 일치 (특수문자는 \로 이스케이프)
```

2. 문자 클래스

```python
문자 클래스 예시
pattern1 = r"[abc]" # a, b, 또는 c 중 하나와 일치
pattern2 = r"[a-z]" # a부터 z까지의 모든 소문자와 일치
pattern3 = r"[A-Z]" # A부터 Z까지의 모든 대문자와 일치
pattern4 = r"[0-9]" # 모든 숫자와 일치 (\d와 동일)
pattern5 = r"[^abc]" # a, b, c를 제외한 모든 문자와 일치
```

3. 미리 정의된 문자 클래스

```python
자주 사용되는 특수 문자 클래스
pattern1 = r"\d" # 숫자 [0-9]
pattern2 = r"\D" # 숫자가 아닌 문자
pattern3 = r"\w" # 단어 문자 [a-zA-Z0-9_]
pattern4 = r"\W" # 단어 문자가 아닌 문자
pattern5 = r"\s" # 공백 문자 (스페이스, 탭, 줄바꿈)
pattern6 = r"\S" # 공백이 아닌 문자
```

4. 수량자

```python
반복 횟수 지정
pattern1 = r"a" # a가 0회 이상 반복
pattern2 = r"a+" # a가 1회 이상 반복
pattern3 = r"a?" # a가 0회 또는 1회 등장
pattern4 = r"a{3}" # a가 정확히 3회 반복
pattern5 = r"a{2,4}" # a가 2회 이상 4회 이하 반복
pattern6 = r"a{2,}" # a가 2회 이상 반복
```

5. 경계 지정자

```python
위치 지정
pattern1 = r"^hello" # 문자열의 시작이 hello
pattern2 = r"world$" # 문자열의 끝이 world
pattern3 = r"\bhello\b" # 단어 경계의 hello
```

6. 그룹화와 참조

```python
그룹 만들기
pattern1 = r"(ab)+" # ab가 1회 이상 반복
pattern2 = r"(?:ab)+" # 캡처하지 않는 그룹
pattern3 = r"(?P<name>ab)+" # 이름이 있는 그룹
```

7. 실제 사용 예시

```python
import re
이메일 주소 매칭
email_pattern = r"[a-zA-Z0-9.%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
한국 전화번호 매칭
phone_pattern = r"01[016789]-?\d{3,4}-?\d{4}"
URL 매칭
url_pattern = r"https?://(?:www\.)?[\w-]+\.[\w.]+/?[\w-.~:/?#\[\]@!$&'()+,;=]"
주요 메서드
text = "연락처: 010-1234-5678"
re.search(pattern, text) # 첫 번째 매칭 찾기
re.findall(pattern, text) # 모든 매칭 찾기
re.sub(pattern, repl, text) # 매칭된 부분 교체
re.split(pattern, text) # 패턴으로 문자열 분할
```

8. 자주 사용되는 패턴 예시

```python
날짜 형식
date_pattern = r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
비밀번호 유효성 검사 (8자 이상, 대소문자, 숫자, 특수문자 포함)
password_pattern = r"^(?=.[A-Z])(?=.[a-z])(?=.\d)(?=.[@$!%?&])[A-Za-z\d@$!%?&]{8,}$"
한글 이름 (2-4자)
korean_name_pattern = r"[가-힣]{2,4}"
IP 주소
ip_pattern = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
```

9. 플래그 옵션

```python
주요 플래그
re.IGNORECASE # 또는 re.I - 대소문자 구분 없이
re.MULTILINE # 또는 re.M - 여러 줄 모드
re.DOTALL # 또는 re.S - .이 줄바꿈도 포함
re.VERBOSE # 또는 re.X - 패턴에 주석 허용
사용 예시
pattern = re.compile(r"""
\d+ # 숫자 부분
[-] # 하이픈
\d+ # 숫자 부분
""", re.VERBOSE)
```

</div>

---

## 2) `json` (JSON 처리)

JSON 형식의 문자열을 다루기 위한 라이브러리
웹에서 데이터 송수신 시 많이 사용

```python
import json

data = {
"name": "Alice",
"age": 25
}
json_str = json.dumps(data) # Python dict -> JSON 문자열
print(json_str) # {"name": "Alice", "age": 25}

parsed_data = json.loads(json_str) # JSON 문자열 -> Python dict
print(parsed_data["name"]) # Alice
```

---

## 3) `csv` (CSV 처리)

CSV 파일(콤마로 구분된 데이터) 읽기/쓰기

```python
import csv

# 쓰기

with open("data.csv", "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(["Name", "Age", "City"])
  writer.writerow(["Alice", 25, "Seoul"])
  writer.writerow(["Bob", 30, "Busan"])

# 읽기

with open("data.csv", "r") as f:
  reader = csv.reader(f)
  for row in reader:
    print(row)
```

---

# 47: (실습2) 표준 라이브러리 활용

[실습2]

현재 디렉토리에 test 폴더를 생성한 뒤, 그 안에 practice.txt 파일을 만들어서 "Hello World"를 작성해 보세요.

힌트: `os.mkdir()`, `open()`, `write()` 등

1초 간격으로 5번 "카운트다운"을 출력한 뒤, "발사!"를 출력하는 코드를 작성해 보세요.

힌트: `time.sleep(1)`

JSON 파일을 하나 생성하여, 간단한 정보(이름, 나이)를 저장한 뒤 다시 불러와서 출력해 보세요.

## 힌트: `json.dump()`, `json.load()`

```python

# 예시 아이디어 (힌트)

# import os, time, json

# # 1) 폴더 만들기

# if not os.path.exists("test"):

# os.mkdir("test")

# # 2) practice.txt 생성 및 쓰기

# with open("test/practice.txt", "w") as f:

# f.write("Hello World")

# # 3) 카운트다운

# for i in range(5, 0, -1):

# print(i)

# time.sleep(1)

# print("발사!")

# # 4) JSON 파일 생성

# data = {"name": "Alice", "age": 25}

# with open("test/data.json", "w") as f:

# json.dump(data, f)

# # 5) JSON 파일 읽기

# with open("test/data.json", "r") as f:

# loaded_data = json.load(f)

# print(loaded_data)

```

---

# 49: 마무리 및 추가 안내

오늘 학습한 내용 요약

- 내장 함수(Built-in Functions) 개요 및 주요 함수
- 표준 라이브러리(standard library)
- 수학/시간(`math`, `random`, `datetime`, `time`)
- 운영체제(`os`, `sys`, `shutil`)
- 문자열/파일포맷(`re`, `json`, `csv`) 등
- 실습: 다양한 라이브러리를 활용하는 방법

<br>
추가 학습 가이드

- 공식 문서에서 라이브러리별 상세 정보 확인 가능
- 더 깊은 활용을 위해, `itertools`, `collections`, `functools` 등도 도전해보세요!
- 외부 라이브러리(`numpy`, `pandas`, `requests`, etc.)도 필요에 따라 학습해보시길 권장합니다.

---

# 50: (강의 종료)

수고하셨습니다!

내장 함수와 표준 라이브러리를 잘 활용하면 코딩이 훨씬 편리해집니다.
다음 시간에는 파이썬 활용 프로젝트, 또는 외부 라이브러리 활용 등 더 심화된 내용을 다룰 예정입니다.

<br>
감사합니다.

---

# 51: 내장 함수 및 표준 라이브러리 치트시트

## 내장 함수 치트시트

| 함수      | 설명                           | 예시 코드                         |
| --------- | ------------------------------ | --------------------------------- |
| `len()`   | 자료형의 길이를 반환           | `len([1, 2, 3])` -> 3             |
| `range()` | 특정 범위의 정수 시퀀스를 생성 | `range(5)` -> [0, 1, 2, 3, 4]     |
| `type()`  | 객체의 자료형을 반환           | `type(3.14)` -> `<class 'float'>` |
| `print()` | 콘솔에 출력                    | `print("Hello")`                  |
| `input()` | 콘솔에서 사용자 입력 받음      | `input("Enter: ")`                |

---

| 함수          | 설명                                            | 예시 코드                                       |
| ------------- | ----------------------------------------------- | ----------------------------------------------- |
| `sum()`       | 합계를 반환                                     | `sum([1, 2, 3])` -> 6                           |
| `min()`       | 최소값을 반환                                   | `min([1, 2, 3])` -> 1                           |
| `max()`       | 최대값을 반환                                   | `max([1, 2, 3])` -> 3                           |
| `abs()`       | 절대값을 반환                                   | `abs(-5)` -> 5                                  |
| `round()`     | 반올림 값을 반환                                | `round(3.14159, 2)` -> 3.14                     |
| `enumerate()` | 인덱스와 값을 함께 반환                         | `enumerate(['a', 'b'])` -> [(0, 'a'), (1, 'b')] |
| `map()`       | 각 요소에 함수를 적용하여 새로운 반복 객체 생성 | `map(str, [1, 2, 3])` -> ['1', '2', '3']        |
| `filter()`    | 조건을 만족하는 요소만 걸러냄                   | `filter(lambda x: x > 0, [-1, 0, 1])` -> [1]    |

---

## 표준 라이브러리 치트시트

<div class="overflow-y-auto max-h-[400px] border rounded p-4">

| 모듈       | 설명                                         | 예시 코드                                            | 슬라이드 이동     |
| ---------- | -------------------------------------------- | ---------------------------------------------------- | ----------------- |
| `math`     | 수학 관련 함수 제공                          | `math.sqrt(16)` -> 4.0                               | [이동](#math)     |
| `random`   | 난수 생성 관련 함수 제공                     | `random.randint(1, 10)` -> 7                         | [이동](#random)   |
| `datetime` | 날짜와 시간 관련 함수 제공                   | `datetime.datetime.now()` -> 현재 날짜와 시간        | [이동](#datetime) |
| `time`     | 시간 관련 함수 제공                          | `time.sleep(2)` -> 2초 대기                          | [이동](#time)     |
| `os`       | 운영체제 기능 제공                           | `os.getcwd()` -> 현재 작업 디렉토리                  | [이동](#os)       |
| `sys`      | 인터프리터 관련 정보 제공                    | `sys.version` -> 파이썬 버전                         | [이동](#sys)      |
| `shutil`   | 파일/디렉토리 복사, 이동 등 고수준 작업 제공 | `shutil.copy("src", "dst")`                          | [이동](#shutil)   |
| `re`       | 정규 표현식 관련 함수 제공                   | `re.findall(r'\d+', 'abc123')` -> ['123']            | [이동](#re)       |
| `json`     | JSON 형식의 문자열 처리                      | `json.dumps({"key": "value"})` -> '{"key": "value"}' | [이동](#json)     |
| `csv`      | CSV 파일 읽기/쓰기                           | `csv.reader(open('file.csv'))`                       | [이동](#csv)      |

</div>

---
