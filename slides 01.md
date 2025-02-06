---
title: "로봇활용 첨단 생산시스템 전문가 양성"

theme: default
---

# Python 기초 다지기

## AI카메라활용 머신러닝

<br>

### 로봇활용 첨단 생산시스템 전문가 양성

<br>

#### 이승현, 2square@korea.ac.kr

<br>

#### 2025-01-21

---

### 0: 강사 소개

<br>

## 이승현

<br>

<div class="grid grid-cols-2 gap-4">
  <div>

### **학력**

- 경북대학교, 위치정보시스템공학과, 공학사 (2017)
- 경북대학교, 융복합시스템공학부, 공학석사 (2019)
- 고려대학교, 컴퓨터학부, 박사 수료 (2021)

<br>

### **기업 경력**

- 2022 - 현재, 블록체인 스타트업, 선임연구원
  - <span style="color:gray">사업 기획, 개발(스마트 컨트랙트, DeFi)</span>
- 2025 - 현재, (주)노바프로토콜, 대표
  - <span style="color:gray">AI Agent 기반의 블록체인 프로그램 감사</span>
  - <span style="color:gray">블록체인 관련 개발 및 컨설팅</span>

</div>
<div>

### **주요 업무**

- 서비스 기획 및 개발, 컨설팅
  - <span style="color:gray">블록체인, 스마트 컨트랙트</span>
- 강의
  - <span style="color:gray">파이썬, 블록체인, AI기초</span>

<br>

### **강의 경력**

- 2025 - 현재, 서울여자대학교 디지털미디어학과 겸임교수
- 2024 - 현재, 건국대학교 메타버스학과 겸임교수
- <span style="color:gray">2020 - 2023, 중앙대 / 이화여자대 / 동국대 강사</span>
- <span style="color:gray">그외 여러 기관 강의</span>

</div>
</div>

---

# 1: 학습 목표

- 파이썬 기초 문법을 이해하고, 간단한 예제를 통해 실습할 수 있다.
- 함수, 제어문, 자료구조, 파일 입출력 등 전반적인 파이썬 활용 능력을 습득한다.
- 객체지향 프로그래밍(OOP)의 기본 개념과 파이썬에서의 구현 방법을 익힌다.
- 파이썬의 표준 라이브러리와 패키지 활용 방법을 배워 간단한 프로젝트를 진행할 수 있다.

---

**커리큘럼 안내**

1. 파이썬 소개, 개발환경 구성
2. 기본 문법(변수, 자료형, 연산자 등)
3. 제어문(조건문, 반복문)
4. 함수(정의, 인자, 반환값)
5. 데이터 구조(리스트, 튜플, 딕셔너리, 세트)
6. 파일 입출력, 예외 처리
7. 객체지향 프로그래밍(클래스, 상속 등)
8. 라이브러리, 패키지 개념 및 표준 라이브러리 소개
9. 파이썬을 활용한 간단 프로젝트 예시
10. 마무리 및 질의응답

---

# 2: 파이썬 소개

파이썬(Python)은 1991년 귀도 반 로섬(Guido van Rossum)이 만든 고급 프로그래밍 언어입니다.

- 문법이 간결하고 직관적이며, 여러 분야(웹, 데이터 과학, AI, 임베디드 등)에서 폭넓게 활용됩니다.
- 초보자와 전문가 모두에게 인기 있는 언어입니다.

<br>

### 파이썬의 장점

- **코드 가독성**: 들여쓰기를 이용해 코드 블록을 구분, 가독성이 뛰어남
- **풍부한 라이브러리**: 데이터 과학, 웹, 게임, GUI 등 다양한 분야의 라이브러리 제공
- **커뮤니티**: 오픈 소스이며, 전 세계에 걸쳐 커뮤니티 활성화
- **생산성**: 개발 속도가 빠르고, 유지보수하기 쉬움

<br>

### 파이썬의 단점

- **실행 속도**: 컴파일 언어(C/C++ 등)에 비해 상대적으로 느릴 수 있음
- **모바일 분야**: 모바일 앱 개발에서는 상대적으로 사용 빈도가 적음

---

# 3: 개발환경 구성

**1. 파이썬 설치**

- [python.org](https://www.python.org/)에서 최신 버전 다운로드 (Windows/Mac/Linux 모두 지원)
- 설치 중 `PATH` 추가 옵션이 있다면 체크 추천
- VSC와 함께 파이썬을 설치하거나, Jupyter Notebook을 설치하는 것을 추천합니다.

**2. 코드 편집기(IDE) 선택**

- VS Code, PyCharm, Jupyter Notebook 등 다양한 IDE 선택 가능
- 초보자라면 VS Code + Python 확장팩이 가볍고 편리
- PyCharm의 경우 실무 개발에 적합
- Jupyter Notebook은 데이터 과학, 머신러닝 등 다양한 분야에서 사용

---

**3. Jupyter Notebook**

- 웹 기반의 대화형 개발 환경
- 데이터 과학, 머신러닝, 교육 등 다양한 분야에서 사용
- 코드, 텍스트, 시각화 결과를 하나의 노트북 파일로 관리 가능
- 설치 방법: `pip install notebook` (VSC 설치 시 자동 설치)
- 실행 방법: 터미널에서 `jupyter notebook` 명령어 입력

---

**4. Google Colab**

- 구글에서 제공하는 클라우드 기반의 Jupyter Notebook 환경
- 별도의 설치 없이 웹 브라우저에서 바로 사용 가능
- GPU, TPU 등 고성능 하드웨어 지원
  - 실무 프로젝트가 아닌 경우, 노트북 구매보다 머신러닝, 딥러닝에 가격 효율적
- 구글 드라이브와 연동하여 파일 관리 가능
- 사용 방법: [Google Colab](https://colab.research.google.com) 접속 후 새 노트북 생성

---

**5. 실습 환경 준비**

- 명령 프롬프트(터미널) 또는 IDE의 터미널 활용

  - 파이썬이 정상 설치되었는지 확인:

  ```bash
  python --version
  ```

- ipynb 파일에서의 실행

---

**6. 가상환경(권장)**

- 프로젝트별 독립적인 환경 유지가 유리
- 예: `venv` 모듈을 이용해 가상환경 생성

```bash
python -m venv myenv
source myenv/bin/activate  # (Linux/Mac)
myenv\Scripts\activate     # (Windows)
```

상세한 내용은 어렵고 파이썬 문법과는 별개의 내용이므로, 추후에 다루겠습니다.

---

# 4: (Q&A)

지금까지의 내용에 대한 질문을 받아봅시다.

- 파이썬 설치 과정 중 궁금한 점
- IDE 선택 관련 질문
- 기타 환경 구성 관련 이슈

> (Q&A 시간은 필요에 따라 유동적으로 운영해 주세요.)

---

# 5: 기본 문법 - 변수와 자료형

## 변수(Variable)

- 데이터를 저장하기 위한 이름표
- 파이썬에서는 별도의 타입 선언 없이, 할당 시점에 타입이 결정됨

<br>

### 자료형(기본 타입)

1. **정수(int)**: 예) `10`, `-3`
2. **실수(float)**: 예) `3.14`, `-1.5`
3. **문자열(str)**: 예) `"Hello"`, `'Python'`
4. **불(bool)**: `True` / `False`

```python
# 변수 예시
my_int = 10
my_float = 3.14
my_str = "Hello Python"
my_bool = True

print(my_int, my_float, my_str, my_bool)
```

---

# 6: 기본 문법 - 연산자

### 산술 연산자

두 개의 값을 연산하는 연산자

- `+`, `-`, `*`, `/`, `//`, `%`, `**`
- `/`는 실수 나눗셈, `//`는 몫, `%`는 나머지, `**`는 거듭제곱

```python
# 연산자 예시
a = 10
b = 3

add = a + b    # 13
sub = a - b    # 7
mul = a * b    # 30
```

---

### 비교 연산자

두 개의 값을 비교하는 연산자

- `==`, `!=`, `>`, `<`, `>=`, `<=`

```python
# 비교 연산자 예시
a = 10
b = 3

equal = a == b    # False
not_equal = a != b    # True
greater = a > b    # True
less = a < b    # False
greater_equal = a >= b    # True
less_equal = a <= b    # False
```

---

### 논리 연산자

두 개의 값을 논리적으로 연산하는 연산자

- `and`, `or`, `not`

```python
# 논리 연산자 예시
a = 10
b = 3

logic_check = (a > b) and (b != 0)  # True
logic_check = (a > b) and (b != 0)  # True
print(add, sub, mul, div, floordiv, mod, exp, logic_check)
```

- `&&`, `||`, `!`

```python
# 논리 연산자 예시
a = 10
b = 3

logic_check = (a > b) && (b != 0)  # True
logic_check = (a > b) || (b == 0)  # True
logic_check = !(a > b)  # False
```

---

# 7: 예제 실습 (기본 문법)

> **[실습1]**

1. 변수 `x`, `y`를 만들고 서로 다른 값(int, float 등)을 할당해 보세요.
2. 두 변수의 산술 연산 결과를 출력해 보세요.
3. 비교 연산자와 논리 연산자를 이용해 참/거짓이 되는 다양한 조합을 만들어 출력해 보세요.

> **[연습문제]**  
> 아래에 원하는 값을 할당하고 결과를 예측해 보세요.

```python
x = 7
y = 2

print(x + y)      # ?
print(x - y)      # ?
print(x / y)      # ?
print(x // y)     # ?
print(x % y)      # ?
print(x > y)      # ?
print((x > y) and (y < 0))  # ?
```

---

# 8: (쉬는 시간)

> 잠깐 쉬는 시간을 갖고, 궁금한 사항은 쉬는 시간 이후에 이어서 질문 받겠습니다.  
> **(쉬는 시간 10분)**

---

# 9: 제어문 - 조건문

### `if` 문 기본 구조

```python
if 조건:
    # 조건이 True일 때 실행
elif 다른_조건:
    # 위 조건이 False이고 이 조건이 True일 때 실행
else:
    # 모든 조건이 False일 때 실행
```

**예시**

```python
score = 85

if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
else:
    grade = 'C'

print("당신의 등급은:", grade)
```

---

# 10: 제어문 - 반복문(while, for)

### `while` 문

```python
count = 0
while count < 5:
    print("Count:", count)
    count += 1
```

### `for` 문

```python
# range(start, end, step)
for i in range(5):
    print("i의 값:", i)

fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

### while, for 비교

```python

```

---

# 11: 예제 실습 (제어문)

> **[실습2]**

1. 점수를 입력받아 (예: `input()` 이용) 90 이상이면 "A", 80 이상이면 "B", 그 외 "C"를 출력하는 프로그램을 작성해 보세요.
2. 1부터 100까지 수의 합을 구하는 코드를 `while` 문과 `for` 문 각각으로 작성해 보세요.

```python
# 실습2 아이디어 (힌트)
# input_score = int(input("점수를 입력하세요: "))
# if 문 사용해서 등급 판별
# while 혹은 for 사용해서 합계 계산
```

---

# 12: 제어문 퀴즈

> **[퀴즈]**

1. `if score >= 60:` 코드를 통해 `60점 이상`인지 확인하려고 할 때, `=`을 하나만 써서 `if score > 60:` 이라고 작성하면 어떻게 될까요?
2. `while True:` 문은 어떤 상황에서 사용해야 하며, 언제 멈춰야 할까요?
3. `for i in range(2, 10):`과 같이 작성할 때 `i` 값은 어떤 범위를 순회하나요?

---

# 13: (Q&A)

> 지금까지 학습한 기본 문법, 제어문에 대해 궁금하신 점이 있나요?  
> (이후 자료구조, 함수로 넘어가기 전 질문받습니다.)

---

# 14: 함수(Function)

**정의**

- 반복적으로 사용되는 코드를 묶어서 사용하기 위한 문법적 도구
- 함수 이름, 매개변수, 반환값으로 구성

### 함수 정의

```python
def 함수이름(매개변수):
    # 실행 코드
    return 결과값  # (생략 가능)
```

### 예시

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # 8
```

---

# 15: 함수의 다양한 형태

1. **매개변수와 반환값이 모두 있는 경우**
2. **매개변수만 있는 경우**
3. **반환값만 있는 경우**
4. **둘 다 없는 경우**

```python
# 매개변수와 반환값이 모두 있는 예
def multiply(x, y):
    return x * y

# 매개변수만 있는 예
def greet(name):
    print("Hello,", name)

# 반환값만 있는 예
def get_pi():
    return 3.14159

# 둘 다 없는 예
def say_hello():
    print("Hello world")
```

---

# 16: 예제 실습 (함수)

> **[실습3]**

1. 두 수를 입력받아, 큰 수를 반환하는 함수를 작성해 보세요.
2. 리스트를 입력받아, 리스트의 모든 원소를 한 줄에 출력만 하는 함수를 작성해 보세요.
3. 함수 호출 후, 결과를 콘솔에 출력하여 정상 동작을 확인해 보세요.

```python
# 힌트
# def get_max(a, b):
#     if a > b:
#         return a
#     else:
#         return b
#
# def print_list(lst):
#     for item in lst:
#         print(item, end=' ')
#     print()  # 개행
```

---

# 17: 데이터 구조 - 리스트(List)

- **리스트**: 순서가 있는 변경 가능한(mutable) 자료형
- 대괄호 `[ ]` 로 생성, 인덱스로 접근
- 슬라이싱(slicing) 지원

```python
my_list = [10, 20, 30, 40]
print(my_list[0])     # 10
my_list[2] = 300
print(my_list)        # [10, 20, 300, 40]

# 슬라이싱
print(my_list[1:3])   # [20, 300]
```

---

# 18: 데이터 구조 - 튜플(Tuple) & 딕셔너리(Dictionary) & 세트(Set)

### 튜플(Tuple)

- 순서가 있지만, 변경이 불가능(immutable)
- 괄호 `( )` 사용

```python
my_tuple = (10, 20, 30)
# my_tuple[1] = 200  # 에러(변경 불가)
```

---

### 딕셔너리(Dictionary)

- 키(key)와 값(value)으로 이루어진 구조
- 중괄호 `{ }` 사용, `{"키": 값}` 형태

```python
person = {
    "name": "Alice",
    "age": 25
}
print(person["name"])  # Alice
person["age"] = 26
```

---

### 세트(Set)

- 중복이 없고 순서가 없음
- 중괄호 `{ }` 사용

```python
my_set = {1, 2, 2, 3}
print(my_set)  # {1, 2, 3}
```

---

# 19: 예제 실습 (데이터 구조)

> **[실습4]**

1. 빈 리스트를 만든 뒤, 사용자로부터 5개의 값을 입력받아 리스트에 추가하고, 최종 리스트를 출력해 보세요.
2. 튜플 2개를 합쳐서 새로운 튜플을 만들어 보세요.
3. 딕셔너리를 이용해, 간단한 전화번호부(`이름: 전화번호`)를 만든 뒤 특정 이름을 통해 번호를 조회해 보세요.
4. 세트를 이용해 중복 제거 예시를 만들어 보세요.

```python
# 힌트
# phone_book = {"홍길동": "010-1234-5678", "이몽룡": "010-9999-0000"}
# print(phone_book["홍길동"])
```

---

# 20: (Q&A)

> 지금까지 배운 데이터 구조(List, Tuple, Dictionary, Set)에 대한 질문을 받습니다.  
> (Q&A 후 이어서 파일 입출력과 예외 처리로 넘어갑니다.)

---

# 21: 파일 입출력

### 파일 다루기의 기본 개념

- 파일은 우리가 컴퓨터에 저장하는 데이터 묶음입니다
- 파이썬으로 파일을 읽고(read) 쓸(write) 수 있습니다

<br>

### 파일 열기/쓰기/닫기의 3단계

1. 파일 열기 - 컴퓨터에게 "이 파일을 사용하겠다"고 알림
2. 파일 읽기/쓰기 - 실제로 파일을 읽거나 내용을 씀
3. 파일 닫기 - 파일 사용이 끝났음을 알림

```python
# 기본적인 방법
f = open("메모.txt", "w")  # 1단계: 파일 열기
f.write("안녕하세요")      # 2단계: 파일에 글쓰기
f.close()                 # 3단계: 파일 닫기
```

---

### 파일 모드 이해하기

파일을 열 때 "모드"를 지정해야 합니다:

- `"r"` - **읽기**(read) 모드: 파일의 내용을 읽을 때
- `"w"` - **쓰기**(write) 모드: 새로운 내용을 쓸 때 (기존 내용 삭제)
- `"a"` - **추가**(append) 모드: 기존 내용 뒤에 새로운 내용을 추가할 때

```python
# 읽기 모드로 파일 열기
f = open("메모.txt", "r")
내용 = f.read()
print(내용)
f.close()
```

---

### 더 안전한 방법: with 구문 사용하기

- `with` 구문을 사용하면 파일을 자동으로 닫아줍니다
- 파일 닫기를 깜빡 잊어도 안전합니다!

```python
# with 구문 사용 예시
with open("메모.txt", "w") as 파일:
    파일.write("안녕하세요\n")
    파일.write("파이썬 공부 중입니다")
# 여기서 자동으로 파일이 닫힙니다!

# 파일 읽기도 같은 방식으로!
with open("메모.txt", "r") as 파일:
    내용 = 파일.read()
    print(내용)
```

---

# 22: 예외 처리

- 프로그램 실행 중 오류가 발생해도 중단되지 않도록 처리
- `try`, `except`, `finally` 구문 사용

```python
try:
    x = int(input("숫자를 입력하세요: "))
    print(10 / x)
except ValueError:
    print("숫자가 아닙니다!")
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다!")
finally:
    print("예외 처리 예제 종료")
```

---

# 23: 예제 실습 (파일 입출력 & 예외 처리)

> **[실습5]**

1. 텍스트 파일(예: `data.txt`)에 3개의 문자열을 저장하는 코드를 작성해 보세요.
2. 해당 파일을 다시 읽어와, 한 줄씩 출력해 보세요.
3. 입력받은 값이 0이면 `ZeroDivisionError`를 발생시키는 코드를 작성해 보고, 예외 처리를 추가해 보세요.

```python
# 힌트
# with open("data.txt", "w") as f:
#     f.write("Line1\n")
#     f.write("Line2\n")
#     f.write("Line3\n")
```

---

# 24: (쉬는 시간)

> **(쉬는 시간 10분)**  
> 지금까지 학습한 내용을 가볍게 복습하거나, 동료와 짧게 이야기를 나눠 보세요.

---

# 25: 객체지향 프로그래밍(OOP) - 기본 개념

Object Oriented Programming

**객체지향 프로그래밍**이란?

<div class="grid grid-cols-2 gap-4">
  <div>

    - 프로그램을 객체(데이터+메서드)의 집합으로 바라보는 패러다임
      - 데이터: 객체가 가지고 있는 속성
        - 예: 사람의 이름, 나이, 주소 등
      - 메서드: 객체가 가지고 있는 동작
        - 예: 사람의 걷기, 말하기, 먹기 등

    - 파이썬은 클래스(class)를 통해 객체를 정의

  </div>
  <div>
  <br>
  <br>

    [Person 클래스 구조]
    +-----------------+
    |     Person      |
    +-----------------+
    | - name: str     |
    | - age: int      |
    +-----------------+
    | + walk():       |
    | + talk():       |
    | + eat():        |
    +-----------------+

  </div>
</div>

---

### 클래스와 객체

- **클래스(Class)**: 객체를 만들기 위한 설계도
- **객체(Object)**: 클래스로부터 생성된 실체(인스턴스)

예시:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"안녕하세요, 저는 {self.name}이고, 나이는 {self.age}살입니다.")

p1 = Person("홍길동", 30)
p1.greet()
```

---

# 26: 클래스 상속(Inheritance)

- 클래스 간에 상하위 관계를 만들어, 상위 클래스(부모 클래스)의 기능을 하위 클래스(자식 클래스)가 물려받는 것

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def eat(self):
        print(self.name, "가 먹이를 먹습니다.")

class Dog(Animal):
    def bark(self):
        print(self.name, "가 멍멍!")

dog = Dog("멍멍이")
dog.eat()   # 부모 클래스 메서드
dog.bark()  # 자식 클래스 메서드
```

---

# 27: 예제 실습 (객체지향)

> **[실습6]**

1. `Student` 클래스를 정의하고, 이름과 학번을 속성으로 가지도록 하세요.
2. `study`라는 메서드를 만들어 "OOO 학생이 공부를 시작합니다."를 출력해 보세요.
3. `Student` 클래스를 상속받는 `GraduateStudent` 클래스를 만들어, "대학원생"임을 알리는 메서드를 추가해 보세요.

```python
# 힌트
# class Student:
#     def __init__(self, name, student_id):
#         self.name = name
#         self.student_id = student_id
#     def study(self):
#         print(f"{self.name} 학생이 공부를 시작합니다.")

# class GraduateStudent(Student):
#     def announce(self):
#         print(f"{self.name}은(는) 대학원생입니다.")
```

---

# 28: (Q&A)

> 객체지향 프로그래밍, 클래스/상속에 대한 질문을 받아봅니다.

---

# 29: 라이브러리 & 패키지 개념

**라이브러리**: 특정 기능을 쉽게 구현할 수 있도록 제공되는 도구들의 집합

### 표준 라이브러리

- 파이썬에 기본적으로 내장된 라이브러리
- 예: `math`, `datetime`, `os`, `sys`, `random` 등

#### 예시: `math` 모듈

```python
import math

print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.141592653589793
```

---

### 외장 라이브러리

- 파이썬 표준 라이브러리 외에도, 다양한 외장 라이브러리가 존재합니다.
- 외장 라이브러리는 `pip` 명령어를 통해 설치할 수 있습니다.

<br>

#### 예시: `requests` 모듈

- HTTP 요청을 보내고 응답을 받을 수 있는 라이브러리

```python
import requests

response = requests.get("https://api.github.com")
print(response.json())
```

---

# **패키지**:

## 관련된 모듈들을 하나의 디렉토리로 묶어 관리하는 방법

<br>

### 패키지 구조

- 패키지는 디렉토리와 `__init__.py` 파일로 구성됩니다.
- `__init__.py` 파일은 해당 디렉토리를 패키지로 인식하게 합니다.

#### 예시: 패키지 구조

```
my_package/
    __init__.py
    module1.py
    module2.py
```

---

# 30: 예제 실습 (라이브러리 활용)

> **[실습7]**

1. `random` 모듈을 이용해 랜덤한 숫자 5개를 뽑아서 리스트로 저장해 보세요.

   - `random.randint(1, 100)`: 1부터 100 사이의 랜덤한 정수를 반환

```python
# 힌트
# import random
# numbers = []
# for _ in range(5):
#     numbers.append(random.randint(1, 100))
# print(numbers)
```

---

2. `datetime` 모듈을 이용해 현재 날짜와 시간을 출력해 보세요.

- `datetime.datetime.now()`: 현재 날짜와 시간을 반환

```python
# 힌트
# import datetime
# now = datetime.datetime.now()
# print(now)
```

---

3. `os` 모듈을 이용해 현재 디렉토리 내 파일 목록을 확인해 보세요.

- `os.listdir(".")`: 현재 디렉토리 내 파일 목록을 반환

```python
# 힌트
# import os
# print(os.listdir("."))
```

---

# 31: 간단한 프로젝트 예시 - "가위바위보" 게임

> **개요**

- 가위바위보 게임 구현
- 사용자가 가위, 바위, 보 중 하나를 선택하면, 컴퓨터도 랜덤으로 선택하여 결과를 출력

  - `input()` 함수를 사용시, 문자열로 사용자 입력 받음

- 컴퓨터는 어떻게 랜덤으로 가위, 바위, 보를 낼 수 있을까요?
- 사용자와 컴퓨터의 선택을 어떻게 비교할 수 있을까요?

---

# 32: "가위바위보" 게임 예제 코드

> (단순 예시)

```python
import random

def rps_game():
    choices = ["가위", "바위", "보"]
    score = {"user": 0, "computer": 0}

    for _ in range(5):
        user_choice = input("가위, 바위, 보 중 하나를 선택하세요: ")
        computer_choice = random.choice(choices)
        print(f"컴퓨터의 선택: {computer_choice}")

        if user_choice == computer_choice:
            print("비겼습니다!")
        elif (user_choice == "가위" and computer_choice == "보") or \
             (user_choice == "바위" and computer_choice == "가위") or \
             (user_choice == "보" and computer_choice == "바위"):
            print("사용자가 이겼습니다!")
            score["user"] += 1
        else:
            print("컴퓨터가 이겼습니다!")
            score["computer"] += 1

```

---

# 33: 프로젝트 실습 과제

> **[실습8]**

1. `rps_game.py` 파일을 생성하고, 가위바위보 게임 코드를 작성해 보세요.
2. 게임을 5번 반복하여 최종 승자를 출력하는 기능을 추가해 보세요.
3. 더 개선해보고 싶다면 다음을 시도해 보세요:
   - 사용자의 승리 횟수와 컴퓨터의 승리 횟수를 저장하고, 최종 결과를 출력하기
   - 게임 결과를 파일에 저장하여 나중에 다시 확인할 수 있도록 하기
   - 어떻게 해야 코드를 더 짧게 작성할 수 있을까요?

---

# 34: (Q&A)

> 프로젝트 진행 중 궁금한 점을 질문받습니다.

---

# 35: 마무리 정리

### 오늘 학습한 내용 요약

1. **파이썬 기초 문법**: 변수, 자료형, 연산자
2. **제어문**: if, while, for
3. **함수**: 정의, 매개변수, 반환값
4. **데이터 구조**: 리스트, 튜플, 딕셔너리, 세트
5. **파일 입출력, 예외 처리**
6. **객체지향 프로그래밍**: 클래스, 상속
7. **라이브러리, 패키지 개념**
8. **표준 라이브러리 활용**
9. **간단한 프로젝트(단어 퀴즈)**

> 위 주제들을 모두 실습하며, 7시간 분량으로 진행했습니다.

---

# 36: 추가 학습 자료

- **공식 문서**: [docs.python.org](https://docs.python.org/3/) (영문)
- **온라인 강의 플랫폼**:
  - 파이썬 무료/유료 강좌 다양
- **국내 개발 커뮤니티**: 파이썬 사용자 모임, 오픈채팅 등
- **개발 관련 유튜브 채널**: 예제, 실습 영상 참고

---

# 37: 최종 질의응답

> 이 강의에서 다룬 내용 및 추가로 궁금한 부분에 대해 Q&A 시간을 가지겠습니다.  
> 실습 내용, 프로젝트 응용, 향후 학습 방향 등에 대해 자유롭게 질문해 주세요.

---

# 38: (강의 종료)

**감사합니다!**




