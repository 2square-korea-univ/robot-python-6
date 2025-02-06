## Slidev 교안 작성

아래 slidev 형식에 맞춰 교안을 작성했습니다.

````markdown
---
# default layout for every page
layout: default
---

## OpenCV-Python 스터디

### 이미지 처리 및 컴퓨터 비전 기초

---

## layout: intro

# OpenCV-Python 스터디

<br>

**이미지 처리 및 컴퓨터 비전**

<br>
<br>

gramman

---

## 목차

- 이미지 다루기
- 영상 다루기
- 도형 그리기
- Mouse로 그리기
- Trackbar
- Basic Operation
- 이미지 연산
- 이미지 Processing
- 이미지 임계처리
- 이미지의 기하학적 변형
- Image Smoothing
- Morphological Transformations
- Image Gradients
- Image Pyramids
- Image Contours
- Contour Feature
- Contour Property
- Contours Hierarchy
- 히스토그램
- 히스토그램 균일화
- 2D Histogram
- 푸리에 변환
- 템플릿 매칭
- 허프 변환
- Hough Circle Transform
- Watershed 알고리즘을 이용한 이미지 분할
- k-Nearest Neighbour(kNN)
- kNN을 이용한 숫자 인식
- Demo 준비 1, 2

---

## 이미지 다루기

---

### Goal

- 이미지 파일을 읽고, 보고, 저장하는 방법 학습
- `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()` 함수 이해

---

### 이미지 읽기

OpenCV 모듈 import:

```python
import cv2
```
````

`cv2.imread()` 함수를 이용하여 이미지 파일 읽기

```python
img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
```

---

### `cv2.imread(fileName, flag)`

이미지 파일을 `flag` 값에 따라 읽어들임

| Parameter  | Type | 설명                                      |
| :--------- | :--- | :---------------------------------------- |
| `fileName` | str  | 이미지 파일 경로                          |
| `flag`     | int  | 이미지 파일을 읽을 때 옵션 (아래 표 참고) |

**Returns**: `image` 객체 행렬 (`numpy.ndarray`)

---

### 이미지 읽기 Flag

| Flag 값                | 설명                                            | 상수 |
| :--------------------- | :---------------------------------------------- | :--- |
| `cv2.IMREAD_COLOR`     | Color 이미지로 읽기 (투명 부분 무시, Default)   | `1`  |
| `cv2.IMREAD_GRAYSCALE` | Grayscale 이미지로 읽기 (이미지 처리 중간 단계) | `0`  |
| `cv2.IMREAD_UNCHANGED` | Alpha channel 포함하여 읽기                     | `-1` |

<br>
<br>

**Note**: Flag 대신 `1`, `0`, `-1` 사용 가능

---

### 이미지 Shape 확인

`img.shape` 를 통해 이미지의 형태 확인 (3차원 행렬)

```python
>>> img.shape
(206, 207, 3)
```

- 206: 행 (Y축)
- 207: 열 (X축)
- 3: BGR 값 (Blue, Green, Red)

<br>
<br>

**Note**: Grayscale 이미지는 2차원 행렬

---

### 이미지 보기

`cv2.imshow()` 함수를 이용하여 이미지 보기

```python
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### `cv2.imshow(title, image)`

윈도우 창에 이미지 표시

| Parameter | Type            | 설명                            |
| :-------- | :-------------- | :------------------------------ |
| `title`   | str             | 윈도우 창 제목                  |
| `image`   | `numpy.ndarray` | `cv2.imread()` 반환 값 (이미지) |

---

### `cv2.waitKey(delay)` & `cv2.destroyAllWindows()`

- `cv2.waitKey(delay)`: 키보드 입력 대기 함수
  - `0`: 키 입력까지 무한 대기
  - `milisecond` 값: 특정 시간 동안 대기
- `cv2.destroyAllWindows()`: 화면의 모든 윈도우 종료

<br>
<br>

**Note**: 일반적으로 `cv2.imshow()`, `cv2.waitKey()`, `cv2.destroyAllWindows()` 함께 사용

---

### Sample Code - 이미지 읽고 보기

```python
import cv2

fname = 'lena.jpg'

original = cv2.imread(fname, cv2.IMREAD_COLOR)
gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
unchange = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

cv2.imshow('Original', original)
cv2.imshow('Gray', gray)
cv2.imshow('Unchange', unchange)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Sample Image - Flag 별 결과

| Original                         | Grayscale                         | Unchange                         |
| :------------------------------- | :-------------------------------- | :------------------------------- |
| ![](images/original_example.png) | ![](images/grayscale_example.png) | ![](images/unchange_example.png) |

---

### 이미지 저장하기

`cv2.imwrite()` 함수를 이용하여 이미지 저장

```python
cv2.imwrite('lenagray.png', gray)
```

---

### `cv2.imwrite(fileName, image)`

이미지 파일을 저장

| Parameter  | Type            | 설명                      |
| :--------- | :-------------- | :------------------------ |
| `fileName` | str             | 저장될 파일 이름          |
| `image`    | `numpy.ndarray` | 저장할 이미지 (`ndarray`) |

---

### Sample Code - 이미지 저장

```python
import cv2

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27: # esc key
    cv2.destroyAllWindows()
elif k == ord('s'): # 's' key
    cv2.imwrite('lenagray.png',img)
    cv2.destroyAllWindows()
```

<br>
<br>

**Warning**: 64bit OS 경우 `k = cv2.waitKey(0) & 0xFF` 로 bit 연산 필요

---

### Matplotlib 사용하기

Matplotlib: Python Plot Library

- 이미지 zoom, 여러 이미지 동시 보기 유용

```python
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)

plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
```

---

### Matplotlib 결과

![](images/matplotlib_bgr.png)

<br>
<br>

**문제점**: 색상이 다르게 출력 (붉은색 -> 파란색)

**이유**: OpenCV는 BGR, Matplotlib는 RGB 사용

---

### Matplotlib RGB 변환

OpenCV BGR -> Matplotlib RGB 변환 필요

```python
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)

b, g, r = cv2.split(img)
img2 = cv2.merge([r,g,b]) # B, R 채널 순서 변경

plt.imshow(img2)
plt.xticks([])
plt.yticks([])
plt.show()
```

---

### Matplotlib RGB 변환 결과

![](images/matplotlib_rgb.png)

<br>
<br>

**해결**: RGB 색상으로 정상 출력

---

## 영상 다루기

---

### Goal

- 동영상 읽기, 재생, 저장 방법 학습
- `cv2.VideoCapture()`, `cv2.VideoWriter()` 함수 이해

---

### Camera로 부터 영상 재생

1. `VideoCapture` 객체 생성 (Camera device index 또는 동영상 파일 경로)
   - Camera 연결: index `0`
2. Loop를 통해 frame 읽기
3. 읽은 frame 변환 및 화면에 표시
4. 영상 재생 종료 후 `VideoCapture` 객체 release, window 닫기

---

### Sample Code - Camera 영상 재생 (Grayscale)

```python
import cv2

cap = cv2.VideoCapture(0) # Camera device index

print('width: {0}, height: {1}'.format(cap.get(3), cap.get(4)))
cap.set(3, 320) # width
cap.set(4, 240) # height

while(True):
    ret, frame = cap.read() # Frame capture

    if (ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Grayscale 변환
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' key 종료
            break

cap.release()
cv2.destroyAllWindows()
```

---

### File로 부터 영상 재생

Camera 영상 재생과 동일

```python
import cv2

cap = cv2.VideoCapture('vtest.avi') # 동영상 파일 경로

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

<br>
<br>

**Note**: 동영상 Codec 설치 필요

---

### 영상 저장

`cv2.VideoWriter` 객체 생성 필요

```python
cv2.VideoWriter(outputFile, fourcc, frame, size)
```

| Parameter    | Type  | 설명                                    |
| :----------- | :---- | :-------------------------------------- |
| `outputFile` | str   | 저장될 파일 이름                        |
| `fourcc`     | int   | Codec 정보 (`cv2.VideoWriter_fourcc()`) |
| `frame`      | float | 초당 저장될 frame 수                    |
| `size`       | tuple | 저장될 사이즈 (width, height)           |

---

### Fourcc Codec 정보

- `cv2.VideoWriter_fourcc('M','J','P','G')` 또는 `cv2.VideoWriter_fourcc(*'MJPG')`
- OS 별 지원 Codec 상이 (Windows: DIVX)

---

### Sample Code - 영상 저장

```python
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640, 480))

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 0) # 이미지 상하 반전
        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

---

## 도형 그리기

---

### Goal

- 다양한 도형 그리기 학습
- `cv2.line()`, `cv2.circle()`, `cv2.rectangle()`, `cv2.putText()` 함수 사용법 숙지

---

### Line 그리기

`cv2.line(img, start, end, color, thickness)`

| Parameter   | Type            | 설명                                |
| :---------- | :-------------- | :---------------------------------- |
| `img`       | `numpy.ndarray` | 그림을 그릴 이미지 파일             |
| `start`     | tuple           | 시작 좌표 (x, y)                    |
| `end`       | tuple           | 종료 좌표 (x, y)                    |
| `color`     | tuple           | BGR 색상 (ex: `(255, 0, 0)` - Blue) |
| `thickness` | int             | 선 두께 (pixel)                     |

---

### Sample Code - Line 그리기

```python
import numpy as np
import cv2

img = np.zeros((512, 512, 3), np.uint8) # Black canvas
img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5) # Blue line

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 사각형 그리기

`cv2.rectangle(img, start, end, color, thickness)`

| Parameter   | Type            | 설명                                   |
| :---------- | :-------------- | :------------------------------------- |
| `img`       | `numpy.ndarray` | 그림을 그릴 이미지 파일                |
| `start`     | tuple           | 시작 좌표 (top-left corner) (x, y)     |
| `end`       | tuple           | 종료 좌표 (bottom-right corner) (x, y) |
| `color`     | tuple           | BGR 색상                               |
| `thickness` | int             | 선 두께 (pixel)                        |

---

### Sample Code - 사각형 그리기

```python
img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3) # Green rectangle
```

---

### 원 그리기

`cv2.circle(img, center, radian, color, thickness)`

| Parameter   | Type            | 설명                                |
| :---------- | :-------------- | :---------------------------------- |
| `img`       | `numpy.ndarray` | 그림을 그릴 이미지 파일             |
| `center`    | tuple           | 원 중심 좌표 (x, y)                 |
| `radian`    | int             | 반지름                              |
| `color`     | tuple           | BGR 색상                            |
| `thickness` | int             | 선 두께 (pixel), `-1`: 원 안쪽 채움 |

---

### Sample Code - 원 그리기

```python
img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1) # Red circle (filled)
```

---

### 타원 그리기

`cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])`

| Parameter    | Type            | 설명                                           |
| :----------- | :-------------- | :--------------------------------------------- |
| `img`        | `numpy.ndarray` | 그림을 그릴 이미지 파일                        |
| `center`     | tuple           | 타원 중심 좌표 (x, y)                          |
| `axes`       | tuple           | 중심에서 가장 큰/작은 거리 (장축, 단축 반지름) |
| `angle`      | float           | 타원 기울기 각도                               |
| `startAngle` | float           | 타원 시작 각도                                 |
| `endAngle`   | float           | 타원 종료 각도                                 |
| `color`      | tuple           | BGR 색상                                       |
| `thickness`  | int             | 선 두께 (pixel), `-1`: 타원 안쪽 채움          |

---

### Sample Code - 타원 그리기

```python
img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1) # White ellipse (half, filled)
```

---

### Polygon 그리기

`cv2.polylines(img, pts, isClosed, color, thickness)`

| Parameter   | Type            | 설명                                  |
| :---------- | :-------------- | :------------------------------------ |
| `img`       | `numpy.ndarray` | 그림을 그릴 이미지 파일               |
| `pts`       | `numpy.ndarray` | 꼭지점 좌표 array (`(N, 1, 2)` shape) |
| `isClosed`  | bool            | 닫힌 도형 여부 (`True` or `False`)    |
| `color`     | tuple           | BGR 색상                              |
| `thickness` | int             | 선 두께 (pixel)                       |

---

### Sample Code - Polygon 그리기

```python
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2)) # 3차원 행렬로 변환
img = cv2.polylines(img, [pts], True, (0, 255, 255)) # Yellow polygon (closed)
```

---

### Text 추가

`cv2.putText(img, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])`

| Parameter   | Type            | 설명                                         |
| :---------- | :-------------- | :------------------------------------------- |
| `img`       | `numpy.ndarray` | 그림을 그릴 이미지 파일                      |
| `text`      | str             | 표시할 문자열                                |
| `org`       | tuple           | 문자열 시작 위치 (bottom-left corner) (x, y) |
| `font`      | int             | Font type (`cv2.FONT_XXX`)                   |
| `fontScale` | float           | Font 크기                                    |
| `color`     | tuple           | Font 색상                                    |

---

### Sample Code - Text 추가

```python
cv2.putText(img, 'OpenCV', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2) # White text
```

---

### 도형 그리기 결과

![](images/drawing_result.png)

---

## Mouse로 그리기

---

### Goal

- Mouse Event 적용 방법 학습
- `cv2.setMouseCallback()` 함수 이해

---

### 작동 방법

OpenCV Mouse Event 종류 확인:

```python
>>> import cv2
>>> events = [i for i in dir(cv2) if 'EVENT' in i]
>>> print(events)
```

<br>
<br>

`cv2.setMouseCallback(windowName, callback, param=None)`

- Mouse Event 확인 및 Callback 호출 함수

---

### `cv2.setMouseCallback(windowName, callback, param=None)`

| Parameter    | Type     | 설명                                           |
| :----------- | :------- | :--------------------------------------------- |
| `windowName` | str      | Window 이름                                    |
| `callback`   | function | Callback 함수 (event, x, y, flags, param 전달) |
| `param`      | any      | Callback 함수에 전달될 Data (optional)         |

---

### 간단한 Demo - Double-Click 원 그리기

```python
import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK: # Double-click event
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1) # Blue circle

img = np.zeros((512, 512, 3), np.uint8) # Black canvas
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(0) & 0xFF == 27: # ESC key 종료
        break

cv2.destroyAllWindows()
```

---

### Advanced Demo - Drag & Draw (사각형/원)

```python
import cv2
import numpy as np

drawing = False # Mouse 클릭 상태 확인
mode = True # True: 사각형, False: 원
ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN: # Mouse 클릭 시작
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE: # Mouse 이동
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1) # Blue rectangle
            else:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1) # Green circle
    elif event == cv2.EVENT_LBUTTONUP: # Mouse 클릭 종료
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1)
        else:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

img = np.zeros((512, 512, 3), np.uint8) # Black canvas
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'): # 'm' key: Mode 변경
        mode = not mode
    elif k == 27: # ESC key: 종료
        break

cv2.destroyAllWindows()
```

---

## Trackbar

---

### Goal

- Trackbar와 OpenCV 연동 방법 학습
- `cv2.getTrackbarPos()`, `cv2.createTrackbar()` 함수 이해

---

### Demo - RGB Color Control

4개의 Trackbar:

- RGB 값 조절 (3개)
- 초기화 (1개)

---

### `cv2.createTrackbar(trackbarName, windowName, value, count, onChange)`

Trackbar 생성 및 Named Window 등록

| Parameter      | Type     | 설명                                                                       |
| :------------- | :------- | :------------------------------------------------------------------------- |
| `trackbarName` | str      | Trackbar 이름                                                              |
| `windowName`   | str      | Trackbar를 등록할 Named Window 이름                                        |
| `value`        | int      | Trackbar 초기 값                                                           |
| `count`        | int      | Trackbar 최대 값 (최소 값: 0)                                              |
| `onChange`     | function | Slide 값 변경 시 호출되는 Callback 함수 (trackbar position parameter 전달) |

---

### `cv2.getTrackbarPos(trackbarName, windowName)`

Trackbar 현재 위치 값 반환

| Parameter      | Type | 설명                                |
| :------------- | :--- | :---------------------------------- |
| `trackbarName` | str  | Trackbar 이름                       |
| `windowName`   | str  | Trackbar가 등록된 Named Window 이름 |

---

### Sample Code - Trackbar Demo (RGB Control)

```python
import cv2
import numpy as np

def nothing(x): # Dummy callback function
    pass

img = np.zeros((300, 512, 3), np.uint8) # Black canvas
cv2.namedWindow('image')

cv2.createTrackbar('R', 'image', 0, 255, nothing) # R Trackbar
cv2.createTrackbar('G', 'image', 0, 255, nothing) # G Trackbar
cv2.createTrackbar('B', 'image', 0, 255, nothing) # B Trackbar

switch = '0:OFF\n1:ON'
cv2.createTrackbar(switch, 'image', 1, 1, nothing) # Switch Trackbar

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27: # ESC key 종료
        break

    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0: # Switch OFF: Black
        img[:] = 0
    else: # Switch ON: RGB Color
        img[:] = [b, g, r]

cv2.destroyAllWindows()
```

---

## Basic Operation

---

### Goal

- Pixel 값 접근 및 수정
- 이미지 기본 속성 확인
- 이미지 ROI (Region of Image) 설정
- 이미지 Channel 분리 및 병합

---

### Pixel Value 접근

이미지 Load 시 3차원 행렬 생성 (행, 열, 색 정보)

```python
import cv2
import numpy as np

img = cv2.imread('lena.jpg')

px = img[100, 200] # (100, 200) pixel 값 접근
print(px) # [157 100 190] (BGR)

b = img[100, 200, 0] # Blue channel 값 접근
print(b) # 157
```

---

### Pixel Value 수정

```python
img[100, 200] = [255, 255, 255] # (100, 200) pixel 흰색 변경

# Numpy 사용 방법
img.item(10, 10, 2) # (10, 10) pixel Red 값 (59)
img.itemset((10, 10, 2), 100) # (10, 10) pixel Red 값 100으로 변경
img.item(10, 10, 2) # (10, 10) pixel Red 값 (100)
```

---

### 이미지 기본 속성

- `img.shape`: 이미지 행, 열, channel 정보 (tuple)
  - Grayscale 이미지: 행, 열 정보만 반환
- `img.size`: 전체 Pixel 수
- `img.dtype`: 이미지 Data type

```python
>>> img.shape
(206, 207, 3)

>>> img.size
42642

>>> img.dtype
dtype('uint8')
```

---

### 이미지 ROI (Region of Image)

ROI 설정: Numpy indexing 사용

```python
img = cv2.imread('baseball-player.jpg')
ball = img[409:454, 817:884] # ROI 설정 (ball 영역)
img[470:515, 817:884] = ball # ROI 영역 Copy
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 이미지 ROI 결과

| Original                                 | Result                                     |
| :--------------------------------------- | :----------------------------------------- |
| ![](images/baseball-player_original.jpg) | ![](images/baseball-player_roi_result.jpg) |

---

### 이미지 Channels

Color 이미지: B, G, R 채널 분리 및 병합

```python
b, g, r = cv2.split(img) # 채널 분리
img = cv2.merge((r, g, b)) # 채널 병합 (RGB 순서 변경)

b = img[:, :, 0] # Blue channel 접근 (Numpy indexing)
```

<br>
<br>

**Warning**: `cv2.split()` 함수는 비용이 많이 드는 함수, Numpy indexing 사용 권장

---

### Channel 값 변경

```python
img[:, :, 2] = 0 # Red channel 값 0으로 변경 (Red 제거 효과)
```

---

## 이미지 연산

---

### Goal

- 이미지 덧셈, Blending, 비트 연산 이해
- `cv2.add()`, `cv2.addWeighted()` 함수 사용법 숙지

---

### 이미지 더하기

- OpenCV `cv2.add()` 함수 vs. Numpy 연산 (`img1 + img2`)
- Saturation 연산 (OpenCV) vs. Modulo 연산 (Numpy)

<br>
<br>

**Note**:

- Saturation 연산: 0 이하 -> 0, 255 이상 -> 255
- Modulo 연산: 256 초과 -> 256으로 나눈 나머지 값

---

### 이미지 더하기 결과 비교

| Original 1                  | Original 2                  | OpenCV Add                        | Numpy Add                        |
| :-------------------------- | :-------------------------- | :-------------------------------- | :------------------------------- |
| ![](images/opencv_add1.png) | ![](images/opencv_add2.png) | ![](images/opencv_add_result.png) | ![](images/numpy_add_result.png) |

---

### 이미지 Blending

이미지 가중치 합 (Blending)

\[ g(x) = (1 - \alpha)f*{0}(x) + \alpha f*{1}(x) \]

- α 값 (0 ~ 1) 변화에 따른 이미지 전환

---

### Sample Code - 이미지 Blending

```python
import cv2
import numpy as np

img1 = cv2.imread('images/flower1.jpg')
img2 = cv2.imread('images/flower2.jpg')

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('W', 'image', 0, 100, nothing)

while True:
    w = cv2.getTrackbarPos('W', 'image')
    dst = cv2.addWeighted(img1, float(100 - w) * 0.01, img2, float(w) * 0.01, 0) # Blending
    cv2.imshow('dst', dst)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
```

---

### 이미지 Blending 결과

![](images/blending_result.gif)

---

### 비트 연산

- AND, OR, NOT, XOR 연산
- 특정 영역 추출 시 유용 (배경 제거, 이미지 합성 등)

---

### Sample Code - 비트 연산 (Logo 합성)

```python
import cv2
import numpy as np

img1 = cv2.imread('images/logo.png') # Logo image
img2 = cv2.imread('images/lena.jpg') # Background image

rows, cols, channels = img1.shape # Logo image shape
roi = img2[0:rows, 0:cols] # Background ROI 설정

img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # Logo grayscale 변환
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY) # Mask 생성 (Logo: 흰색, 배경: 검은색)
mask_inv = cv2.bitwise_not(mask) # Inverse mask (Logo: 검은색, 배경: 흰색)

img1_fg = cv2.bitwise_and(img1, img1, mask=mask) # Logo foreground 추출 (배경 제거)
img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv) # Background background 추출 (Logo 영역 제거)

dst = cv2.add(img1_fg, img2_bg) # Logo foreground + Background background 합성
img2[0:rows, 0:cols] = dst # 합성 이미지 원본 이미지에 추가

cv2.imshow('res', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 비트 연산 결과 (Logo 합성)

![](images/bit_operation_result.png)

---

## 이미지 임계처리

---

### Goal

- 이미지 이진화 방법 학습
  - Simple thresholding
  - Adaptive thresholding
  - Otsu’s thresholding
- `cv2.threshold()`, `cv2.adaptiveThreshold()` 함수 사용법 숙지

---

### 기본 임계처리 (Simple Thresholding)

- 고정 임계값 사용
- `cv2.threshold()` 함수 사용

```python
cv2.threshold(src, thresh, maxval, type)
```

| Parameter | Type            | 설명                                    |
| :-------- | :-------------- | :-------------------------------------- |
| `src`     | `numpy.ndarray` | Input image (single-channel, grayscale) |
| `thresh`  | int             | 임계값                                  |
| `maxval`  | int             | 임계값 초과 시 적용할 Value             |
| `type`    | int             | Thresholding type (아래 표 참고)        |

---

### Thresholding Type

| Type 상수               | 설명                                               |
| :---------------------- | :------------------------------------------------- |
| `cv2.THRESH_BINARY`     | Threshold 값보다 크면 `maxval`, 작으면 0           |
| `cv2.THRESH_BINARY_INV` | `cv2.THRESH_BINARY` 반대                           |
| `cv2.THRESH_TRUNC`      | Threshold 값보다 크면 Threshold 값, 작으면 원래 값 |
| `cv2.THRESH_TOZERO`     | Threshold 값보다 크면 원래 값, 작으면 0            |
| `cv2.THRESH_TOZERO_INV` | `cv2.THRESH_TOZERO` 반대                           |

---

### Sample Code - 기본 임계처리

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gradient.jpg', 0) # Grayscale 이미지 로드

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### 기본 임계처리 결과

![](images/thresholding_types.png)

---

### 적응 임계처리 (Adaptive Thresholding)

- 이미지 영역별 임계값 자동 결정
- `cv2.adaptiveThreshold()` 함수 사용

```python
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
```

| Parameter        | Type            | 설명                                                          |
| :--------------- | :-------------- | :------------------------------------------------------------ |
| `src`            | `numpy.ndarray` | Grayscale 이미지                                              |
| `maxValue`       | int             | 임계값 초과 시 적용할 Value                                   |
| `adaptiveMethod` | int             | Threshold value 계산 방법 (아래 표 참고)                      |
| `thresholdType`  | int             | Threshold type (`cv2.THRESH_BINARY`, `cv2.THRESH_BINARY_INV`) |
| `blockSize`      | int             | Thresholding 적용 영역 Size (홀수)                            |
| `C`              | int             | 평균 또는 가중평균 값에서 뺄 값                               |

---

### Adaptive Method

| Method 상수                      | 설명                                                    |
| :------------------------------- | :------------------------------------------------------ |
| `cv2.ADAPTIVE_THRESH_MEAN_C`     | 주변 영역 평균값으로 Threshold value 결정               |
| `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` | 주변 영역 가우시안 가중 평균값으로 Threshold value 결정 |

---

### Sample Code - 적응 임계처리

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/dave.png', 0) # Grayscale 이미지 로드

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # Global thresholding

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2) # Mean adaptive thresholding
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2) # Gaussian adaptive thresholding

titles = ['Original', 'Global', 'Mean', 'Gaussian']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### 적응 임계처리 결과

![](images/adaptive_thresholding_result.png)

---

### Otsu의 이진화 (Otsu's Binarization)

- Bimodal image (히스토그램 Peak 2개) 에서 임계값 자동 계산
- `cv2.threshold()` 함수 flag에 `cv2.THRESH_OTSU` 추가, 임계값 `0` 전달

---

### Sample Code - Otsu 이진화

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/noise.png', 0) # Noisy image 로드

ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # Global thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu thresholding

blur = cv2.GaussianBlur(img, (5, 5), 0) # Gaussian blur (Noise 제거)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Gaussian blur + Otsu thresholding

images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Otsu 이진화 결과

![](images/otsu_thresholding_result.png)

---

## 이미지의 기하학적 변형

---

### Goal

- 기하학적 변형 이해
- `cv2.getPerspectiveTransform()` 함수 사용법 숙지

---

### Transformations (변환)

좌표 `x` -> 좌표 `x'` 변환 함수

- 종류: Scaling, Translation, Rotation 등

변환 종류 분류:

- 강체 변환 (Rigid-Body): 크기, 각도 보존 (Translation, Rotation)
- 유사 변환 (Similarity): 각도 보존, 크기 변화 (Scaling)
- 선형 변환 (Linear): Vector 공간 이동 (이동 변환 제외)
- Affine: 선형 변환 + 이동 변환, 수평성 유지 (사각형 -> 평행사변형)
- Perspective: Affine 변환 + 수평성 유지 X (원근 변환)

---

### Scaling (크기 변환)

- 이미지 크기 변경
- `cv2.resize()` 함수 사용
- 보간법 (Interpolation method) 사용

```python
cv2.resize(img, dsize, fx, fy, interpolation)
```

| Parameter       | Type            | 설명                                                                                |
| :-------------- | :-------------- | :---------------------------------------------------------------------------------- |
| `img`           | `numpy.ndarray` | Input Image                                                                         |
| `dsize`         | tuple           | Manual Size (width, height)                                                         |
| `fx`            | float           | 가로 사이즈 배수 (ex: `2`: 2배 확대, `0.5`: 1/2 축소)                               |
| `fy`            | float           | 세로 사이즈 배수                                                                    |
| `interpolation` | int             | 보간법 (사이즈 축소: `cv2.INTER_AREA`, 확대: `cv2.INTER_CUBIC`, `cv2.INTER_LINEAR`) |

---

### Sample Code - Scaling

```python
import cv2
import numpy as np

img = cv2.imread('images/logo.png')

height, width = img.shape[:2] # 이미지 높이, 너비

shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) # 이미지 축소
zoom1 = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC) # Manual Size 지정 확대
zoom2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) # 배수 Size 지정 확대

cv2.imshow('Original', img)
cv2.imshow('Shrink', shrink)
cv2.imshow('Zoom1', zoom1)
cv2.imshow('Zoom2', zoom2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Scaling 결과

| Original                         | Shrink                         | Zoom 1                        | Zoom 2                        |
| :------------------------------- | :----------------------------- | :---------------------------- | :---------------------------- |
| ![](images/scaling_original.png) | ![](images/scaling_shrink.png) | ![](images/scaling_zoom1.png) | ![](images/scaling_zoom2.png) |

---

### Translation (이동 변환)

- 이미지 위치 변경
- `cv2.warpAffine()` 함수 사용

```python
cv2.warpAffine(src, M, dsize)
```

| Parameter | Type            | 설명                              |
| :-------- | :-------------- | :-------------------------------- |
| `src`     | `numpy.ndarray` | Input Image                       |
| `M`       | `numpy.ndarray` | 변환 행렬 (2x3, float32)          |
| `dsize`   | tuple           | Output image size (width, height) |

<br>
<br>

**Warning**: `dsize` = (width, height) = (columns, rows)

---

### 변환 행렬 (Translation)

2x3 변환 행렬 (float32 numpy array)

\[ M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix} \]

- \( t_x \): X축 이동 거리
- \( t_y \): Y축 이동 거리

---

### Sample Code - Translation

```python
import cv2
import numpy as np

img = cv2.imread('images/logo.png')

rows, cols = img.shape[:2] # 이미지 높이, 너비

M = np.float32([[1, 0, 10], [0, 1, 20]]) # 변환 행렬 (X축 +10, Y축 +20 이동)
dst = cv2.warpAffine(img, M, (cols, rows)) # Translation 적용

cv2.imshow('Original', img)
cv2.imshow('Translation', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Translation 결과

| Original                             | Translation                        |
| :----------------------------------- | :--------------------------------- |
| ![](images/translation_original.png) | ![](images/translation_result.png) |

---

### Rotation (회전 변환)

- 이미지 중심 기준 회전
- `cv2.getRotationMatrix2D()` 함수로 변환 행렬 생성
- `cv2.warpAffine()` 함수로 Rotation 적용

```python
cv2.getRotationMatrix2D(center, angle, scale)
```

| Parameter | Type  | 설명                               |
| :-------- | :---- | :--------------------------------- |
| `center`  | tuple | 이미지 중심 좌표 (x, y)            |
| `angle`   | float | 회전 각도 (양수: 시계 반대 방향)   |
| `scale`   | float | Scale factor (ex: `0.5`: 1/2 크기) |

---

### Sample Code - Rotation

```python
import cv2

img = cv2.imread('images/logo.png')

rows, cols = img.shape[:2] # 이미지 높이, 너비

M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 0.5) # 회전 변환 행렬 (중심 기준 90도 회전, 1/2 크기)
dst = cv2.warpAffine(img, M, (cols, rows)) # Rotation 적용

cv2.imshow('Original', img)
cv2.imshow('Rotation', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Rotation 결과

| Original                          | Rotation                        |
| :-------------------------------- | :------------------------------ |
| ![](images/rotation_original.png) | ![](images/rotation_result.png) |

---

### Affine Transformation (어파인 변환)

- 선의 평행성 유지 (이동, 확대/축소, 회전, 반전 포함)
- 3개의 대응점 필요

---

### Sample Code - Affine Transformation

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/chessboard.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[200, 100], [400, 100], [200, 200]]) # 원본 이미지 좌표
pts2 = np.float32([[200, 300], [400, 200], [200, 400]]) # 변환 후 좌표

cv2.circle(img, (200, 100), 10, (255, 0, 0), -1) # 좌표 표시 (Blue)
cv2.circle(img, (400, 100), 10, (0, 255, 0), -1) # 좌표 표시 (Green)
cv2.circle(img, (200, 200), 10, (0, 0, 255), -1) # 좌표 표시 (Red)

M = cv2.getAffineTransform(pts1, pts2) # Affine 변환 행렬 계산
dst = cv2.warpAffine(img, M, (cols, rows)) # Affine 변환 적용

plt.subplot(121), plt.imshow(img), plt.title('image')
plt.subplot(122), plt.imshow(dst), plt.title('Affine')
plt.show()
```

---

### Affine Transformation 결과

| Image                           | Affine                        |
| :------------------------------ | :---------------------------- |
| ![](images/affine_original.png) | ![](images/affine_result.png) |

---

### Perspective Transformation (원근 변환)

- 선의 평행성 유지 X (원근 효과)
- 4개의 대응점 필요

---

### Sample Code - Perspective Transformation

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/perspective.jpg')

pts1 = np.float32([[504, 1003], [243, 1525], [1000, 1000], [1280, 1685]]) # 원본 이미지 좌표 (좌상->좌하->우상->우하)
pts2 = np.float32([[10, 10], [10, 1000], [1000, 10], [1000, 1000]]) # 변환 후 좌표

cv2.circle(img, (504, 1003), 20, (255, 0, 0), -1) # 좌표 표시 (Blue)
cv2.circle(img, (243, 1524), 20, (0, 255, 0), -1) # 좌표 표시 (Green)
cv2.circle(img, (1000, 1000), 20, (0, 0, 255), -1) # 좌표 표시 (Red)
cv2.circle(img, (1280, 1685), 20, (0, 0, 0), -1) # 좌표 표시 (Black)

M = cv2.getPerspectiveTransform(pts1, pts2) # Perspective 변환 행렬 계산
dst = cv2.warpPerspective(img, M, (1100, 1100)) # Perspective 변환 적용

plt.subplot(121), plt.imshow(img), plt.title('image')
plt.subplot(122), plt.imshow(dst), plt.title('Perspective')
plt.show()
```

---

### Perspective Transformation 결과

| Image                                | Perspective                        |
| :----------------------------------- | :--------------------------------- |
| ![](images/perspective_original.jpg) | ![](images/perspective_result.png) |

---

## Image Smoothing (이미지 스무딩)

---

### Goal

- 다양한 Filter 이용 Blur 이미지 생성
- 사용자 정의 Filter 적용

---

### Image Filtering (이미지 필터링)

- 이미지 주파수 표현:
  - 고주파: 밝기 변화 多 (경계선)
  - 저주파: 밝기 변화 少 (배경)
- Low-pass filter (LPF): 노이즈 제거, Blur 처리
- High-pass filter (HPF): 경계선 검출
- `cv2.filter2D()` 함수: Kernel (filter) 적용

---

### Kernel (필터)

- 행렬 형태
- Kernel 크기 ↑ -> Blur 효과 ↑
- 평균 필터 Kernel 예시 (5x5):

\[ K = \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix} \]

---

### Filter 적용 방법

1. 이미지 각 Pixel에 Kernel 적용
2. Kernel 영역 Pixel 값 Sum 계산
3. Sum 값 Kernel 요소 개수 (ex: 5x5 Kernel = 25) 로 나눔 (평균값 계산)
4. 평균값을 해당 Pixel에 적용

---

### Sample Code - User-defined Filter

```python
import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread('images/lena.jpg')

cv2.namedWindow('image')
cv2.createTrackbar('K', 'image', 1, 20, nothing) # Kernel size trackbar

while(1):
    if cv2.waitKey(1) & 0xFF == 27:
        break
    k = cv2.getTrackbarPos('K', 'image')
    if k == 0: # Kernel size 0 방지
        k = 1
    kernel = np.ones((k, k), np.float32) / (k * k) # Kernel 생성
    dst = cv2.filter2D(img, -1, kernel) # Filter 적용
    cv2.imshow('image', dst)

cv2.destroyAllWindows()
```

---

### User-defined Filter 결과 (5x5 Kernel)

![](images/user_filter_5x5.png)

---

### Image Blurring (이미지 블러링)

- Low-pass filter (LPF) 적용 -> 고주파 영역 제거
- 노이즈 제거, 경계선 흐리게 효과
- OpenCV 블러링 방법 4가지:
  - Averaging
  - Gaussian Filtering
  - Median Filtering
  - Bilateral Filtering

---

### Averaging (평균 블러링)

- Box 형태 Kernel 적용 후 평균값 중심점에 적용
- `cv2.blur()` 또는 `cv2.boxFilter()` 함수 사용
- 3x3 필터 예시:

\[ K = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix} \]

```python
cv2.blur(src, ksize)
```

| Parameter | Type            | 설명                                                                          |
| :-------- | :-------------- | :---------------------------------------------------------------------------- |
| `src`     | `numpy.ndarray` | Input image (channel 수 상관 X, depth: CV_8U, CV_16U, CV_16S, CV_32F, CV_64F) |
| `ksize`   | tuple           | Kernel size (width, height)                                                   |

---

### Data Type 상수

OpenCV 이미지 Data Type 상수:

- `CV_8U`: 8-bit unsigned integer (0~255)
- `CV_8S`: 8-bit signed integer (-128~127)
- `CV_16U`: 16-bit unsigned integer (0~65535)
- `CV_16S`: 16-bit signed integer (-32768~32767)
- `CV_32S`: 32-bit signed integer (-2147483648~2147483647)
- `CV_32F`: 32-bit floating-point number
- `CV_64F`: 64-bit floating-point number

<br>
<br>

**Note**: 채널 수와 함께 표현 (ex: `CV_8UC1` - 8bit unsigned integer, 1 channel)

---

### Gaussian Filtering (가우시안 필터링)

- 가우시안 함수 이용 Kernel 적용
- Kernel 행렬 값 가우시안 함수 통해 수학적 생성
- Kernel size: 양수 & 홀수 지정
- Gaussian Noise (백색 노이즈) 제거 효과적

```python
cv2.GaussianBlur(img, ksize, sigmaX)
```

| Parameter | Type            | 설명                                                                          |
| :-------- | :-------------- | :---------------------------------------------------------------------------- |
| `img`     | `numpy.ndarray` | Input image (channel 수 상관 X, depth: CV_8U, CV_16U, CV_16S, CV_32F, CV_64F) |
| `ksize`   | tuple           | Kernel size (width, height) (양수 & 홀수)                                     |
| `sigmaX`  | float           | X 방향 가우시안 Kernel 표준 편차                                              |

---

### Median Filtering (미디언 필터링)

- Kernel window 영역 Pixel 값 정렬 후 중간값 선택 적용
- Salt-and-pepper noise 제거 효과적

```python
cv2.medianBlur(src, ksize)
```

| Parameter | Type            | 설명                                                                                                  |
| :-------- | :-------------- | :---------------------------------------------------------------------------------------------------- |
| `src`     | `numpy.ndarray` | Input image (1, 3, 4 channel, depth: CV_8U, CV_16U, CV_32F, ksize: 3 or 5, CV_8U: ksize 더 크게 가능) |
| `ksize`   | int             | Kernel size (1보다 큰 홀수)                                                                           |

---

### Bilateral Filtering (양방향 필터링)

- 경계선 유지 Gaussian Blur
- Gaussian 필터 + 주변 Pixel 고려 Gaussian 필터 추가 적용

```python
cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
```

| Parameter    | Type            | 설명                                   |
| :----------- | :-------------- | :------------------------------------- |
| `src`        | `numpy.ndarray` | Input image (8-bit, 1 or 3 channel)    |
| `d`          | int             | 필터링 시 고려할 주변 Pixel 지름       |
| `sigmaColor` | float           | Color 고려 공간 (값 ↑: 먼 색상도 고려) |
| `sigmaSpace` | float           | 공간 고려 공간 (값 ↑: 먼 Pixel도 고려) |

---

### Sample Code - 블러링 비교

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/lena.jpg')

b, g, r = cv2.split(img) # BGR -> RGB (Matplotlib 출력)
img = cv2.merge([r, g, b])

dst1 = cv2.blur(img, (7, 7)) # Averaging blur
dst2 = cv2.GaussianBlur(img, (5, 5), 0) # Gaussian blur
dst3 = cv2.medianBlur(img, 9) # Median blur
dst4 = cv2.bilateralFilter(img, 9, 75, 75) # Bilateral filtering

images = [img, dst1, dst2, dst3, dst4]
titles = ['Original', 'Blur(7X7)', 'Gaussian Blur(5X5)', 'Median Blur', 'Bilateral']

for i in range(5):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i]), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### 블러링 비교 결과

![](images/blurring_comparison.png)

<br>
<br>

**Note**: Gaussian vs. Bilateral - 윤곽선 유지 차이 확인

---

## Morphological Transformations (형태학적 변환)

---

### Goal

- Morphological 변환 (Erosion, Dilation, Opening, Closing) 이해
- `cv2.erode()`, `cv2.dilate()`, `cv2.morphologyEx()` 함수 사용법 숙지

---

### Theory (이론)

- 이미지 Segmentation, 단순화, 보정 -> 형태 파악 목적
- Binary 또는 Grayscale 이미지 적용
- 변환 종류:
  - Dilation (팽창)
  - Erosion (침식)
  - Opening (열림)
  - Closing (닫힘)
- Input: 원본 이미지, Structuring Element (Kernel)

---

### Structuring Element (Kernel)

- 원본 이미지에 적용 Kernel
- 중심점 기준 or 변경 가능
- 주로 사용 형태: 사각형, 타원형, 십자형

---

### Erosion (침식)

- Structuring Element 적용, 하나라도 0 있으면 대상 Pixel 제거
- 작은 Object 제거 효과

![](images/erosion_example.png)
(출처: KOCW)

---

### `cv2.erode(src, kernel, iterations)`

| Parameter    | Type            | 설명                                                          |
| :----------- | :-------------- | :------------------------------------------------------------ |
| `src`        | `numpy.ndarray` | Input image (depth: CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)    |
| `kernel`     | `numpy.ndarray` | Structuring element (`cv2.getStructuringElement()` 함수 생성) |
| `iterations` | int             | Erosion 적용 반복 횟수                                        |

---

### Dilation (팽창)

- Erosion 반대, 대상 확장 및 작은 구멍 채우기
- Structuring Element 적용, 하나라도 겹치면 이미지 확장
- 경계 부드럽게, 구멍 메꿈 효과

![](images/dilation_example.png)
(출처: KOCW)

---

### `cv2.dilate(src, kernel, iterations)`

| Parameter    | Type            | 설명                                                          |
| :----------- | :-------------- | :------------------------------------------------------------ |
| `src`        | `numpy.ndarray` | Input image (depth: CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)    |
| `kernel`     | `numpy.ndarray` | Structuring element (`cv2.getStructuringElement()` 함수 생성) |
| `iterations` | int             | Dilation 적용 반복 횟수                                       |

---

### Opening & Closing

- Erosion & Dilation 조합
- 순서 차이

- Opening: Erosion -> Dilation, 작은 Object/돌기 제거
- Closing: Dilation -> Erosion, 전체 윤곽 파악

```python
cv2.morphologyEx(src, op, kernel, iterations)
```

| Parameter    | Type            | 설명                                                          |
| :----------- | :-------------- | :------------------------------------------------------------ |
| `src`        | `numpy.ndarray` | Input image (depth: CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)    |
| `op`         | int             | Morphological operation type (아래 표 참고)                   |
| `kernel`     | `numpy.ndarray` | Structuring element (`cv2.getStructuringElement()` 함수 생성) |
| `iterations` | int             | Erosion & Dilation 적용 반복 횟수                             |

---

### Morphological Operation Type

| Operation 상수       | 설명                                        |
| :------------------- | :------------------------------------------ |
| `cv2.MORPH_OPEN`     | Opening 연산                                |
| `cv2.MORPH_CLOSE`    | Closing 연산                                |
| `cv2.MORPH_GRADIENT` | Morphological Gradient (Dilation - Erosion) |
| `cv2.MORPH_TOPHAT`   | Top Hat (Original - Opening)                |
| `cv2.MORPH_BLACKHAT` | Black Hat (Closing - Original)              |

---

### Structuring Element 생성

- 사각형: Numpy 이용

```python
import numpy as np
kernel = np.ones((5, 5), np.uint8)
```

- 원, 타원, 십자형: `cv2.getStructuringElement()` 함수 사용

```python
cv2.getStructuringElement(shape, ksize)
```

| Parameter | Type  | 설명                                     |
| :-------- | :---- | :--------------------------------------- |
| `shape`   | int   | Element 모양 (아래 표 참고)              |
| `ksize`   | tuple | Structuring element size (width, height) |

---

### Structuring Element Shape

| Shape 상수          | 설명      |
| :------------------ | :-------- |
| `cv2.MORPH_RECT`    | 사각형    |
| `cv2.MORPH_ELLIPSE` | 타원형    |
| `cv2.MORPH_CROSS`   | 십자 모양 |

---

### Sample Code - Morphological 변환 비교

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

dotImage = cv2.imread('images/dot_image.png')
holeImage = cv2.imread('images/hole_image.png')
orig = cv2.imread('images/morph_origin.png')

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 사각형 Kernel

erosion = cv2.erode(dotImage, kernel, iterations=1) # Erosion
dilation = cv2.dilate(holeImage, kernel, iterations=1) # Dilation

opening = cv2.morphologyEx(dotImage, cv2.MORPH_OPEN, kernel) # Opening
closing = cv2.morphologyEx(holeImage, cv2.MORPH_CLOSE, kernel) # Closing
gradient = cv2.morphologyEx(orig, cv2.MORPH_GRADIENT, kernel) # Gradient
tophat = cv2.morphologyEx(orig, cv2.MORPH_TOPHAT, kernel) # Tophat
blackhat = cv2.morphologyEx(orig, cv2.MORPH_BLACKHAT, kernel) # Blackhat

images = [dotImage, erosion, opening, holeImage, dilation, closing, gradient, tophat, blackhat]
titles = ['Dot Image', 'Erosion', 'Opening', 'Hole Image', 'Dilation', 'Closing', 'Gradient', 'Tophat', 'Blackhot']

for i in range(9):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i]), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Morphological 변환 결과

![](images/morphological_transformations_result.png)

---

## Image Gradients (이미지 기울기)

---

### Goal

- Edge Detection 이해
- Sobel & Scharr, Laplacian, Canny Edge Detection

---

### Gradient (기울기)

- 스칼라장에서 최대 증가율 벡터장 (방향 & 힘)
- 영상 처리: Edge 및 방향 검출 활용
- Pixel (x, y) 벡터값 (크기 & 방향) -> Edge 근접 정도, 방향 파악

---

### Sobel & Scharr Filter

- Gaussian smoothing + 미분
- 노이즈 이미지에 효과적
- X축/Y축 미분 -> 경계값 계산
- 직선 미분 -> 상수, 곡선 미분 -> 방정식 -> Edge 선 표현

---

### `cv2.Sobel(src, ddepth, dx, dy, ksize)`

| Parameter | Type            | 설명                                          |
| :-------- | :-------------- | :-------------------------------------------- |
| `src`     | `numpy.ndarray` | Input image                                   |
| `ddepth`  | int             | Output image depth (`-1`: Input image와 동일) |
| `dx`      | int             | X축 미분 차수                                 |
| `dy`      | int             | Y축 미분 차수                                 |
| `ksize`   | int             | Kernel size (ksize x ksize)                   |

<br>
<br>

**Note**: `ksize = -1` -> 3x3 Scharr filter 적용 (Sobel 3x3 보다 나은 결과)

---

### `cv2.Scharr(src, ddepth, dx, dy)`

- `cv2.Sobel()` 함수와 유사
- `ksize` 자동 적용 (Sobel 3x3 보다 정확)

---

### Laplacian 함수

- 이미지 가로/세로 Gradient 2차 미분 값
- Sobel filter + 미분 정도 추가 (dx, dy = 2 Sobel과 유사)
- Blob (주변 Pixel과 차이 큰 덩어리) 검출 활용

```python
cv2.Laplacian(src, ddepth, ksize)
```

| Parameter | Type            | 설명                        |
| :-------- | :-------------- | :-------------------------- |
| `src`     | `numpy.ndarray` | Input image                 |
| `ddepth`  | int             | Output image depth          |
| `ksize`   | int             | Kernel size (ksize x ksize) |

---

### Canny Edge Detection

- 가장 유명한 Edge Detection 방법
- 다단계 알고리즘:

1. **Noise Reduction**: 5x5 Gaussian filter Noise 제거
2. **Edge Gradient Detection**: Gradient 방향 & 강도 확인 (경계값 후보군 선별)
3. **Non-maximum Suppression**: Edge 아닌 Pixel 제거
4. **Hysteresis Thresholding**: Edge 후보군 -> 진짜 Edge 판별 (강한 Edge, 약한 Edge 분류 후 연결성 확인)

---

### `cv2.Canny(image, threshold1, threshold2, apertureSize, L2gradient)`

| Parameter      | Type            | 설명                                                           |
| :------------- | :-------------- | :------------------------------------------------------------- |
| `image`        | `numpy.ndarray` | Input image (8-bit)                                            |
| `threshold1`   | int             | Hysteresis Thresholding min 값                                 |
| `threshold2`   | int             | Hysteresis Thresholding max 값                                 |
| `apertureSize` | int             | Sobel 연산 Kernel size                                         |
| `L2gradient`   | bool            | Gradient 계산 정확도 (`True`: 정확도 높음, `False`: 속도 빠름) |

---

### Sample Code - Edge Detection 비교

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/dave.png')
canny = cv2.Canny(img, 30, 70) # Canny Edge Detection

laplacian = cv2.Laplacian(img, cv2.CV_8U) # Laplacian Edge Detection
sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3) # Sobel X Edge Detection
sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3) # Sobel Y Edge Detection

images = [img, laplacian, sobelx, sobely, canny]
titles = ['Original', 'Laplacian', 'Sobel X', 'Sobel Y', 'Canny']

for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i]), plt.title([titles[i]])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Edge Detection 비교 결과

![](images/edge_detection_comparison.png)

---

## Image Pyramids (이미지 피라미드)

---

### Goal

- Image Pyramid 이해
- `cv2.pyrUp()`, `cv2.pyrDown()` 함수 사용법 숙지

---

### Theory (이론)

- 동일 이미지 다양한 사이즈 Set
- 얼굴 인식 등 다양한 크기 대상 검출 시 유용
- 종류:
  - Gaussian Pyramids
  - Laplacian Pyramids

---

### Gaussian Pyramid

- High Level (저해상도) -> Lower Level (고해상도) Row/Column 연속 제거 생성
- M x N 이미지 -> M/2 x N/2 (1/4 사이즈 축소)

---

### Sample Code - Gaussian Pyramid

```python
import cv2

img = cv2.imread('images/lena.jpg')

lower_reso = cv2.pyrDown(img) # 1/4 사이즈 축소
higher_reso = cv2.pyrUp(img) # 4배 사이즈 확대

cv2.imshow('img', img)
cv2.imshow('lower', lower_reso)
cv2.imshow('higher', higher_reso)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Gaussian Pyramid 결과

![](images/gaussian_pyramid_result.png)

---

### Laplacian Pyramid

- Gaussian Pyramid 기반 생성
- `cv2.pyrDown()`, `cv2.pyrUp()` 함수 사용 시 원본 이미지 복원 X (해상도 손실)
- 원본 이미지 - (pyrUp(pyrDown(원본 이미지))) -> 외곽선 추출

---

### Sample Code - Laplacian Pyramid

```python
import cv2
import numpy as np

img = cv2.imread('lena.jpg')

GAD = cv2.pyrDown(img) # Gaussian Down
GAU = cv2.pyrUp(GAD) # Gaussian Up
temp = cv2.resize(GAU, (img.shape[1], img.shape[0])) # Resize to original shape
res = cv2.subtract(img, temp) # Laplacian Pyramid 이미지 (외곽선)

cv2.imshow('Laplacian', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Laplacian Pyramid 결과

![](images/laplacian_pyramid_result.png)

---

### 이미지 Blending with Pyramid

Image Pyramid 이용 이미지 Blending 자연스럽게 처리 가능

1. 2개 이미지 Load
2. 각 이미지 Gaussian Pyramid 생성
3. Gaussian Pyramid -> Laplacian Pyramid 생성
4. 각 단계 Laplacian Pyramid 좌/우 결합
5. 결합 결과 확대 -> 동일 사이즈 결합 결과 Add (외곽선 선명)

---

### Sample Code - 이미지 Blending with Pyramid

```python
import cv2
import numpy as np

A = cv2.imread('images/apple.jpg')
B = cv2.imread('images/orange.jpg')

# Gaussian Pyramid 생성 (A 이미지)
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# Gaussian Pyramid 생성 (B 이미지)
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# Laplacian Pyramid 생성 (A 이미지)
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    temp = cv2.resize(gpA[i - 1], (GE.shape[1], GE.shape[0]))
    L = cv2.subtract(temp, GE)
    lpA.append(L)

# Laplacian Pyramid 생성 (B 이미지)
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    temp = cv2.resize(gpB[i - 1], (GE.shape[1], GE.shape[0]))
    L = cv2.subtract(temp, GE)
    lpB.append(L)

# Laplacian Pyramid 단계별 좌/우 결합
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

# Pyramid Blending 결과 복원
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    temp = cv2.resize(LS[i], (ls_.shape[1], ls_.shape[0]))
    ls_ = cv2.add(ls_, temp)

real = np.hstack((A[:, :cols // 2], B[:, cols // 2:])) # 원본 이미지 단순 결합

cv2.imshow('Real Blending', real)
cv2.imshow('Pyramid Blending', ls_)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Pyramid Blending 결과

![](images/pyramid_blending_step5.png)

<br>
<br>

**Note**: 5단계 (확대 및 외곽선 추가) 역할 - 이미지 선명도 향상

---

### Pyramid Blending 최종 결과 비교

| Pyramid Blending                        | Real Blending                        |
| :-------------------------------------- | :----------------------------------- |
| ![](images/pyramid_blending_result.png) | ![](images/real_blending_result.png) |

<br>
<br>

**Note**: Pyramid Blending - 경계면 부드럽게 처리

---

## Image Contours (이미지 윤곽선)

---

### Goal

- Contours (윤곽선) 이해
- `cv2.findContours()`, `cv2.drawContours()` 함수 사용법 숙지

---

### Contours (윤곽선)

- 동일 색상/강도 영역 경계선 연결 선
- 등고선, 일기 예보 등고선 유사
- 대상 외형 파악 유용

![](images/contours_line_example.png)
(출처: 위키피디아)

---

### Contours 특징

- Binary Image 사용 (Thresholding, Canny Edge 선처리)
- `cv2.findContours()`: 원본 이미지 직접 수정 -> 복사본 사용 권장
- 검은 배경 흰색 대상 찾기 (대상: 흰색, 배경: 검은색)

---

### Find & Draw Contours

- `cv2.findContours()`: Contours 찾기
- `cv2.drawContours()`: Contours 그리기

```python
cv2.findContours(image, mode, method)
```

| Parameter | Type            | 설명                                            |
| :-------- | :-------------- | :---------------------------------------------- |
| `image`   | `numpy.ndarray` | Input image (8-bit single-channel binary image) |
| `mode`    | int             | Contours 검색 Mode (아래 표 참고)               |
| `method`  | int             | Contours 근사 방법 (아래 표 참고)               |

**Returns**: `image`, `contours`, `hierarchy`

---

### Contours Mode

| Mode 상수           | 설명                                                 |
| :------------------ | :--------------------------------------------------- |
| `cv2.RETR_EXTERNAL` | 바깥쪽 Contours Line만 검색                          |
| `cv2.RETR_LIST`     | 모든 Contours Line 검색, Hierarchy 관계 구성 X       |
| `cv2.RETR_CCOMP`    | 모든 Contours Line 검색, 2-Level Hierarchy 관계 구성 |
| `cv2.RETR_TREE`     | 모든 Contours Line 검색, 모든 Hierarchy 관계 구성    |

---

### Contours Method

| Method 상수                  | 설명                                                              |
| :--------------------------- | :---------------------------------------------------------------- |
| `cv2.CHAIN_APPROX_NONE`      | 모든 Contours Point 저장                                          |
| `cv2.CHAIN_APPROX_SIMPLE`    | Contours Line 그릴 수 있는 Point만 저장 (ex: 사각형 -> 4개 Point) |
| `cv2.CHAIN_APPROX_TC89_L1`   | Contours Point 검색 알고리즘                                      |
| `cv2.CHAIN_APPROX_TC89_KCOS` | Contours Point 검색 알고리즘                                      |

---

### Method 비교

사각형 Contours Line 그리기

- `cv2.CHAIN_APPROX_NONE`: 모든 Point 저장 -> 메모리 ↑
- `cv2.CHAIN_APPROX_SIMPLE`: 4개 Point만 저장 -> 메모리 ↓

```python
>>> contours[0].shape # cv2.CHAIN_APPROX_SIMPLE (4 point)
(4, 1, 2)

>>> contours[0].shape # cv2.CHAIN_APPROX_NONE (750 point)
(750, 1, 2)
```

---

### `cv2.drawContours(image, contours, contourIdx, color, thickness)`

| Parameter    | Type            | 설명                                                                |
| :----------- | :-------------- | :------------------------------------------------------------------ |
| `image`      | `numpy.ndarray` | 원본 이미지                                                         |
| `contours`   | list            | Contours 정보 (`cv2.findContours()` 반환 값)                        |
| `contourIdx` | int             | Contours List index (그릴 Contours Line index, `-1`: 전체 Contours) |
| `color`      | tuple           | Contours Line 색상                                                  |
| `thickness`  | int             | Contours Line 두께 (pixel, `-1`: Contours 내부 채움)                |

**Returns**: Contours Line 그려진 이미지

---

### Sample Code - Contours 찾고 그리기

```python
import cv2
import numpy as np

img = cv2.imread('images/rectangle.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale 변환

ret, thresh = cv2.threshold(imgray, 127, 255, 0) # Binary Image 변환

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Contours 검색
image = cv2.drawContours(img, contours, -1, (0, 255, 0), 3) # Contours 그리기 (Green)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### Contours 그리기 결과

![](images/contours_drawing_result.png)

---

## Contour Feature (Contour 특징)

---

### Goal

- Contours 특징 (영역, 중심점, Bounding Box 등) 추출
- Contours 특징 추출 함수 학습

---

### Moments (모멘트)

- 대상 구분 특징 (영역, 둘레, 중심점 등)
- 대상 설명 (Describe) 자료 활용

```python
import cv2
import numpy as np

img = cv2.imread('images/rectangle.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0] # 첫번째 Contours
M = cv2.moments(cnt) # Moment 특징 추출

print(M.items()) # Moment 정보 출력
```

---

### Moments 정보

```
[('mu02', 35950058.66666663), ('mu03', 1.52587890625e-05), ..., ('m21', 236404615552.0)]
```

- Dictionary Data (24개 특징 정보)
- 중심점 (cx, cy) 계산 예시:

```python
cx = int(M['m10'] / M['m00']) # 중심점 x 좌표
cy = int(M['m01'] / M['m00']) # 중심점 y 좌표
```

---

### Contour Area (Contour 면적)

- Moments `m00` 값 또는 `cv2.contourArea()` 함수 이용

```python
cv2.contourArea(cnt) # Contour 면적 계산
```

---

### Contour Perimeter (Contour 둘레)

- `cv2.arcLength()` 함수 이용
- 2번째 Parameter: `True` (폐곡선), `False` (개방 곡선)

```python
cv2.arcLength(cnt, True) # 폐곡선 Contours 둘레 길이 계산
cv2.arcLength(cnt, False) # 개방 곡선 Contours 둘레 길이 계산
```

---

### Contour Approximation (Contour 근사화)

- Contours Point 수 감소 -> 근사 Line 생성
- Douglas-Peucker algorithm 사용
- `cv2.approxPolyDP()` 함수 사용

```python
cv2.approxPolyDP(curve, epsilon, closed)
```

| Parameter | Type            | 설명                                                          |
| :-------- | :-------------- | :------------------------------------------------------------ |
| `curve`   | `numpy.ndarray` | Contours Point array                                          |
| `epsilon` | float           | Original Curve vs. 근사 Curve 최대 거리 (값 ↑: Point 수 감소) |
| `closed`  | bool            | 폐곡선 여부                                                   |

**Returns**: 근사화 Contours Point array

---

### Sample Code - Contour Approximation

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/bad_rect.png')
img1 = img.copy()
img2 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0] # 첫 번째 Contours

epsilon1 = 0.01 * cv2.arcLength(cnt, True) # 1% epsilon
epsilon2 = 0.1 * cv2.arcLength(cnt, True) # 10% epsilon

approx1 = cv2.approxPolyDP(cnt, epsilon1, True) # 1% 근사화
approx2 = cv2.approxPolyDP(cnt, epsilon2, True) # 10% 근사화

cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3) # Original Contours (215 Points)
cv2.drawContours(img1, [approx1], 0, (0, 255, 0), 3) # 1% 근사화 Contours (21 Points)
cv2.drawContours(img2, [approx2], 0, (0, 255, 0), 3) # 10% 근사화 Contours (4 Points)

titles = ['Original', '1%', '10%']
images = [img, img1, img2]

for i in range(3):
    plt.subplot(1, 3, i + 1), plt.title(titles[i]), plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Contour Approximation 결과

![](images/contour_approximation_result.png)

---

### Convex Hull (볼록 껍질)

- Contours Point 모두 포함 볼록 외관선
- Contour Approximation 유사 결과, 방법 상이
- convexity defect: Contours vs. Hull 최대 차이

![](images/convex_hull_example.png)

---

### Sample Code - Convex Hull

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/hand.png')
img1 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[1] # 손 모양 Contours
hull = cv2.convexHull(cnt) # Convex Hull 계산

cv2.drawContours(img1, [hull], 0, (0, 255, 0), 3) # Convex Hull 그리기 (Green)

titles = ['Original', 'Convex Hull']
images = [img, img1]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.title(titles[i]), plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Convex Hull 결과

| Original                             | Convex Hull                        |
| :----------------------------------- | :--------------------------------- |
| ![](images/convex_hull_original.png) | ![](images/convex_hull_result.png) |

---

### Checking Convexity (Convex 여부 확인)

- `cv2.isContourConvex()` 함수: Contour Convex 여부 판단 (`True` or `False` 반환)
- Convex: 볼록 or 평평 (오목 부분 X)

```python
>>> cv2.isContourConvex(contours[0]) # 외곽선 Contours (사각형)
True
>>> cv2.isContourConvex(contours[1]) # 손 모양 Contours
False
```

---

### Bounding Rectangle (Bounding 사각형)

Contours Line 둘러싸는 사각형 그리기

1. Straight Bounding Rectangle: 회전 무시 사각형

```python
x, y, w, h = cv2.boundingRect(cnt) # Bounding Rectangle 좌표, 크기 계산
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # 사각형 그리기 (Green)
```

2. Rotated Rectangle: 최소 영역 사각형

```python
rect = cv2.minAreaRect(cnt) # Rotated Rectangle 정보 계산
box = cv2.boxPoints(rect) # Rotated Rectangle 꼭지점 좌표 계산
box = np.int0(box)
img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2) # 사각형 그리기 (Blue)
```

---

### Minimum Enclosing Circle (최소 외접원)

Contours Line 완전히 포함 최소 원 그리기

```python
(x, y), radius = cv2.minEnclosingCircle(cnt) # 최소 외접원 정보 계산
center = (int(x), int(y))
radius = int(radius)
img = cv2.circle(img, center, radius, (0, 255, 0), 2) # 원 그리기 (Green)
```

---

### Fitting an Ellipse (타원 Fitting)

Contours Line 둘러싸는 타원 그리기

```python
ellipse = cv2.fitEllipse(cnt) # 타원 Fitting
img = cv2.ellipse(img, ellipse, (0, 255, 0), 2) # 타원 그리기 (Green)
```

---

### Sample Code - Contour Fitting 비교

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/lightning.jpg')
img1 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[1] # 번개 모양 Contours

x, y, w, h = cv2.boundingRect(cnt) # Straight Rectangle
img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 3) # Green

rect = cv2.minAreaRect(cnt) # Rotated Rectangle
box = cv2.boxPoints(rect)
box = np.int0(box)
img1 = cv2.drawContours(img1, [box], 0, (0, 0, 255), 3) # Blue

(x, y), radius = cv2.minEnclosingCircle(cnt) # Minimum Enclosing Circle
center = (int(x), int(y))
radius = int(radius)
img1 = cv2.circle(img1, center, radius, (255, 255, 0), 3) # Yellow

ellipse = cv2.fitEllipse(cnt) # Ellipse Fitting
img1 = cv2.ellipse(img1, ellipse, (255, 0, 0), 3) # Red

titles = ['Original', 'Result']
images = [img, img1]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.title(titles[i]), plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Contour Fitting 비교 결과

![](images/contour_fitting_result.png)

---

## Contour Property (Contour 속성)

---

### Goal

- Contour 속성 (Aspect Ratio, Extend, Solidity, Extream Points) 이해

---

### Aspect Ratio (가로 세로 비율)

\[ Aspect Ratio = \frac{Width}{Height} \]

```python
x, y, w, h = cv2.boundingRect(cnt) # Bounding Rectangle 정보
aspect_ratio = float(w) / h # Aspect Ratio 계산
```

---

### Extend (Extend 비율)

Contour 면적 vs. Bounding Rectangle 면적 비율

\[ Extend = \frac{Object Area}{Bounding Rectangle Area} \]

```python
area = cv2.contourArea(cnt) # Contour 면적
x, y, w, h = cv2.boundingRect(cnt) # Bounding Rectangle 정보
rect_area = w * h # Bounding Rectangle 면적
extend = float(area) / rect_area # Extend 비율 계산
```

---

### Solidity (Solidity 비율)

Contour 면적 vs. Convex Hull 면적 비율

\[ Solidity = \frac{Contour Area}{Convex Hull Area} \]

```python
area = cv2.contourArea(cnt) # Contour 면적
hull = cv2.convexHull(cnt) # Convex Hull
hull_area = cv2.contourArea(hull) # Convex Hull 면적
solidity = float(area) / hull_area # Solidity 비율 계산
```

---

### Extream Points (극점)

Contour Line 좌우상하 끝점

```python
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0]) # 좌측 끝점
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0]) # 우측 끝점
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0]) # 상단 끝점
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0]) # 하단 끝점
```

<br>
<br>

**Note**: `cnt[:, :, 0]`: Contours Point x 좌표 배열, `argmin()`: 최소값 index 반환

---

### Sample Code - Extream Points

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/UK.jpg')
img1 = img.copy()

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 125, 255, 0)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[14] # UK 지도 Contours

leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0]) # 좌측 끝점
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0]) # 우측 끝점
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0]) # 상단 끝점
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0]) # 하단 끝점

cv2.circle(img1, leftmost, 20, (0, 0, 255), -1) # 좌측 끝점 표시 (Red)
cv2.circle(img1, rightmost, 20, (0, 0, 255), -1) # 우측 끝점 표시 (Red)
cv2.circle(img1, topmost, 20, (0, 0, 255), -1) # 상단 끝점 표시 (Red)
cv2.circle(img1, bottommost, 20, (0, 0, 255), -1) # 하단 끝점 표시 (Red)

img1 = cv2.drawContours(img1, cnt, -1, (255, 0, 0), 5) # Contours 그리기 (Blue)

titles = ['Original', 'Result']
images = [img, img1]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.title(titles[i]), plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Extream Points 결과

| Original                                | Result                                |
| :-------------------------------------- | :------------------------------------ |
| ![](images/extream_points_original.jpg) | ![](images/extream_points_result.png) |

---

## Contours Hierarchy (Contour 계층 구조)

---

### Goal

- Contours Hierarchy 구조 이해
- Contours Mode (RETR_LIST, RETR_EXTERNAL, RETR_CCOMP, RETR_TREE) 별 Hierarchy 정보 확인

---

### Hierarchy (계층 구조)

- 이미지 Contours 간 포함 관계
- 이전 (Next), 이후 (Prev), 부모 (Parent), 자식 (Child) 관계
- `cv2.findContours()` Mode parameter 따라 Hierarchy 정보 결정

---

### Sample Image - Contours Hierarchy

![](images/image_hierarchy_example.png)

---

### Sample Code - Contours Hierarchy

```python
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

img = cv2.imread('images/imageHierarchy.png')

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 125, 255, 0)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    b = random.randrange(1, 255) # Random color 생성
    g = random.randrange(1, 255)
    r = random.randrange(1, 255)

    cnt = contours[i]
    img = cv2.drawContours(img, [cnt], -1, (b, g, r), 2) # Contours 그리기 (Random color)

titles = ['Result']
images = [img]

for i in range(1):
    plt.subplot(1, 1, i + 1), plt.title(titles[i]), plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Contours Hierarchy 결과

![](images/contours_hierarchy_result.png)

<br>
<br>

**Note**: 3, 3a & 4, 4a - Child Contours 분리 -> 포함 관계 표현

---

### `RETR_LIST` Mode

- Hierarchy Shape: `(1, x, 4)`
- 3번째 차원 4개 값: `(next, prev, child, parent)`
- 선/후 관계만 표현, 부모/자식 관계 표현 X
- `child`, `parent` 값: `-1` (대상 X)

```python
>>> hierarchy # RETR_LIST Mode
array([[[ 1, -1, -1, -1],
        [ 2,  0, -1, -1],
        [ 3,  1, -1, -1],
        [ 4,  2, -1, -1],
        [ 5,  3, -1, -1],
        [ 6,  4, -1, -1],
        [ 7,  5, -1, -1],
        [ 8,  6, -1, -1],
        [-1,  7, -1, -1]]])
```

---

### `RETR_EXTERNAL` Mode

- 가장 바깥쪽 Contours Line (Parent X) 만 반환
- 부모/자식 관계 구성 X

```python
>>> hierarchy # RETR_EXTERNAL Mode
array([[[ 1, -1, -1, -1],
        [ 2,  0, -1, -1],
        [-1,  1, -1, -1]]])
```

---

### `RETR_CCOMP` Mode

- 2-Level Hierarchy 표현
  - 바깥쪽 (외곽선): Level 1
  - 안쪽 포함: Level 2

![](images/contours_ccomp_hierarchy.png)

---

### `RETR_CCOMP` Hierarchy 정보

```python
>>> hierarchy # RETR_CCOMP Mode
array([[[ 3, -1,  1, -1],
        [ 2, -1, -1,  0],
        [-1,  1, -1,  0],
        [ 5,  0,  4, -1],
        [-1, -1, -1,  3],
        [ 7,  3,  6, -1],
        [-1, -1, -1,  5],
        [ 8,  5, -1, -1],
        [-1,  7, -1, -1]]])
```

---

### `RETR_TREE` Mode

- 완전한 Hierarchy 표현
  - 누구에게도 포함 X: Level 0
  - 안쪽 포함 Contours: 순차적 Level 부여

![](images/contours_tree_hierarchy.png)

---

### `RETR_TREE` Hierarchy 정보

```python
>>> hierarchy # RETR_TREE Mode
array([[[ 7, -1,  1, -1],
        [-1, -1,  2,  0],
        [-1, -1,  3,  1],
        [-1, -1,  4,  2],
        [-1, -1,  5,  3],
        [ 6, -1, -1,  4],
        [-1,  5, -1,  4],
        [ 8,  0, -1, -1],
        [-1,  7, -1, -1]]])
```

---

## 히스토그램 (Histogram)

---

### Goal

- OpenCV 이용 Histogram 검색
- OpenCV & Matplotlib 이용 Histogram 표현
- `cv2.calcHist()`, `np.histogram()` 함수 사용법 숙지

---

### Histogram (히스토그램)

- 이미지 밝기 분포 그래프 표현
- 이미지 전체 밝기 분포, 채도 파악

![](images/histogram_example.png)
(출처: Cambridgeincolor in Color)

<br>
<br>

**Note**: X축: 색 강도 (0~255), Y축: 해당 색상 Pixel 개수

---

### 히스토그램 관련 용어

- **BINS**: 히스토그램 X축 간격 (위 그림: 256)
  - `histSize` (OpenCV)
- **DIMS**: 분석 대상 값 (빛 강도, RGB 값 등)
- **RANGE**: 측정 값 범위 (X축 from ~ to)

---

### Histogram in OpenCV

`cv2.calcHist()` 함수: Histogram 분석

```python
cv2.calcHist(images, channels, mask, histSize, ranges)
```

| Parameter  | Type            | 설명                                                                      |
| :--------- | :-------------- | :------------------------------------------------------------------------ |
| `images`   | `numpy.ndarray` | 분석 대상 이미지 (uint8 or float32 type, Array 형태)                      |
| `channels` | list            | 분석 채널 (X축 대상) (Grayscale: `[0]`, Color: `[0]`, `[1]`, `[2]` (BGR)) |
| `mask`     | `numpy.ndarray` | 분석 영역 (None: 전체 영역)                                               |
| `histSize` | list            | BINS 값 (ex: `[256]`)                                                     |
| `ranges`   | list            | Range 값 (ex: `[0, 256]`)                                                 |

---

### Sample Code - Grayscale Histogram

```python
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

img1 = cv2.imread('images/flower1.jpg', 0) # Grayscale 이미지 로드
img2 = cv2.imread('images/flower2.jpg', 0) # Grayscale 이미지 로드

hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256]) # Histogram 계산 (Red Line)
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256]) # Histogram 계산 (Green Line)

plt.subplot(221), plt.imshow(img1, 'gray'), plt.title('Red Line')
plt.subplot(222), plt.imshow(img2, 'gray'), plt.title('Green Line')
plt.subplot(223), plt.plot(hist1, color='r'), plt.plot(hist2, color='g') # Histogram Plot (Red, Green)
plt.xlim([0, 256])
plt.show()
```

---

### Grayscale Histogram 결과

![](images/grayscale_histogram_result.png)

<br>
<br>

**Note**: Red Line (어두운 이미지) -> 좌측 분포 ↑, Green Line (밝은 이미지) -> 우측 분포 ↑

---

### Mask 적용 Histogram

- 특정 영역 Histogram 분석: Mask 적용

---

### Sample Code - Mask 적용 Histogram

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/lena.png')

mask = np.zeros(img.shape[:2], np.uint8) # Mask 생성 (검은색)
mask[100:300, 100:400] = 255 # Mask 영역 설정 (흰색 사각형)

masked_img = cv2.bitwise_and(img, img, mask=mask) # Mask 적용 이미지

hist_full = cv2.calcHist([img], [1], None, [256], [0, 256]) # 원본 이미지 Histogram (Green Channel)
hist_mask = cv2.calcHist([img], [1], mask, [256], [0, 256]) # Mask 적용 이미지 Histogram (Green Channel)

plt.subplot(221), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.subplot(222), plt.imshow(mask, 'gray'), plt.title('Mask')
plt.subplot(223), plt.imshow(masked_img, 'gray'), plt.title('Masked Image')
plt.subplot(224), plt.title('Histogram')
plt.plot(hist_full, color='r'), plt.plot(hist_mask, color='b') # Histogram Plot (Red: 원본, Blue: Mask 적용)
plt.xlim([0, 256])

plt.show()
```

---

### Mask 적용 Histogram 결과

![](images/mask_histogram_result.png)

---

## 히스토그램 균일화 (Histogram Equalization)

---

### Goal

- Histogram Equalization 이해
- 이미지 Contrast 향상

---

### Theory (이론)

- 특정 영역 집중 Histogram -> 낮은 Contrast -> 이미지 품질 ↓
- 전체 영역 균등 분포 Histogram -> 좋은 이미지 품질 ↑
- Histogram Equalization: Histogram 균등 분포 변환

![](images/histogram_equalization_theory.png)

---

### Histogram Equalization 방법

1. 각 Pixel Cumulative Distribution Function (CDF) 값 계산
2. Histogram Equalization 공식 대입 -> 0~255 값 변환
3. 변환 값으로 이미지 표현 -> 균일화 이미지 획득

<br>
<br>

**Note**: 자세한 내용 - Wikipedia 참고

---

### Sample Code - Numpy Histogram Equalization

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/hist_unequ.jpg') # Contrast 낮은 이미지 로드

hist, bins = np.histogram(img.flatten(), 256, [0, 256]) # Histogram 계산
cdf = hist.cumsum() # CDF 계산

cdf_m = np.ma.masked_equal(cdf, 0) # CDF == 0 부분 Mask 처리
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) # Histogram Equalization 공식 적용
cdf = np.ma.filled(cdf_m, 0).astype('uint8') # Mask 처리 부분 0으로 채우기

img2 = cdf[img] # Equalization 이미지 생성

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(img2), plt.title('Equalization')
plt.show()
```

---

### Numpy Histogram Equalization 결과

![](images/numpy_histogram_equalization_result.png)

---

### OpenCV Histogram Equalization

`cv2.equalizeHist()` 함수 사용 -> 간편하게 Equalization 처리

```python
import cv2
import numpy as np

img = cv2.imread('images/hist_unequ.jpg', 0) # Grayscale 이미지 로드

img2 = cv2.equalizeHist(img) # Histogram Equalization 적용

img = cv2.resize(img, (400, 400)) # 이미지 Resize (화면 출력 편의)
img2 = cv2.resize(img2, (400, 400))

dst = np.hstack((img, img2)) # 이미지 Horizontal Stack
cv2.imshow('img', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

---

### OpenCV Histogram Equalization 결과

![](images/opencv_histogram_equalization_result.png)

---

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

- Adaptive Histogram Equalization (AHE): 이미지 영역별 Equalization 적용 (문제점: 노이즈 증폭)
- CLAHE: Contrast Limit 적용 AHE (노이즈 증폭 방지)

---

### CLAHE 문제점 해결

- Adaptive Histogram Equalization (AHE) 문제점: 작은 영역 노이즈 증폭
- Contrast Limit 적용: Contrast Limit 초과 영역 -> 다른 영역 균등 배분

---

### Sample Code - CLAHE

```python
import cv2
import numpy as np

img = cv2.imread('images/clahe.png', 0) # Contrast 낮은 이미지 로드

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE 객체 생성 (Contrast Limit: 2, Tile Size: 8x8)
img2 = clahe.apply(img) # CLAHE 적용

img = cv2.resize(img, (400, 400)) # 이미지 Resize (화면 출력 편의)
img2 = cv2.resize(img2, (400, 400))

dst = np.hstack((img, img2)) # 이미지 Horizontal Stack
cv2.imshow('img', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

---

### CLAHE 결과

![](images/clahe_result.png)

<br>
<br>

**Note**: 이미지 윤곽선 유지, 전체 Contrast 향상

---

## 2D Histogram

---

### Goal

- 2D Histogram 검색 및 Plotting

---

### 소개

- 1D Histogram: Grayscale 이미지 Pixel 강도 분석 (빛 세기)
- 2D Histogram: Color 이미지 Hue & Saturation 동시 분석

<br>
<br>

**Note**: Histogram Back-Projection 활용

---

### 적용

1. 이미지 HSV Format 변환
2. `cv2.calcHist()` 함수 적용

```python
cv2.calcHist([image], [channel], mask, [bins], [range])
```

| Parameter | Type            | 설명                                               |
| :-------- | :-------------- | :------------------------------------------------- |
| `image`   | `numpy.ndarray` | HSV 변환 이미지                                    |
| `channel` | list            | 분석 채널 (Hue: `0`, Saturation: `1`)              |
| `bins`    | list            | BINS 값 (Hue: `180`, Saturation: `256`)            |
| `range`   | list            | Range 값 (Hue: `[0, 180]`, Saturation: `[0, 256]`) |

---

### Sample Image - 2D Histogram 분석 대상

![](images/home.jpg)

---

### Sample Code - 2D Histogram

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/home.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGR -> HSV 변환

hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 2D Histogram 계산 (Hue, Saturation)

cv2.imshow('Hist', hist) # OpenCV 이미지 출력 (확인용)
plt.imshow(hist, interpolation='nearest') # Matplotlib 이미지 출력 (Plot)
plt.show()
```

---

### 2D Histogram 결과

![](images/2d_histogram_result.png)

<br>
<br>

**Note**: X축: Saturation, Y축: Hue, Y축 값 집중 (Hue: 100 -> 하늘색, 25 -> 노란색) -> 이미지 하늘색, 노란색 분포 多

---

## 푸리에 변환 (Fourier Transform)

---

### Goal

- Numpy, OpenCV 이용 푸리에 변환 검색
- 푸리에 변환 이용 이미지 변환

---

### 푸리에 변환 (Fourier Transform)

- 주파수 분석 방법
- 시간 도메인 -> 주파수 도메인 변환
- 시간 축 제거, 전체 특징 파악 용이

![](images/fourier_transform_theory.png)
(출처: Incodom)

---

### 푸리에 변환 - 이미지 적용

- 이미지 -> 파동 변환
  - 밝기 변화 多 (경계선): 고주파
  - 밝기 변화 少 (배경): 저주파
- 고주파 제거: Blur 처리
- 저주파 제거: 경계선 추출
- 이미지 -> 푸리에 변환 -> 주파수 필터링 -> 역변환 -> 이미지 가공

---

### Sample Code - Numpy 푸리에 변환 (저주파 제거)

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/lena.jpg', cv2.IMREAD_GRAYSCALE) # Grayscale 이미지 로드

f = np.fft.fft2(img) # 푸리에 변환
fshift = np.fft.fftshift(f) # 저주파 중앙 이동
magnitude_spectrum = 20 * np.log(np.abs(fshift)) # Spectrum 계산 (Log Scaling)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2 # 이미지 중심 좌표

d = 10 # 마스크 크기
fshift[crow - d:crow + d, ccol - d:ccol + d] = 0 # 마스크 생성 (중심 저주파 제거)

f_ishift = np.fft.ifftshift(fshift) # Inverse Shift
img_back = np.fft.ifft2(f_ishift) # 역 푸리에 변환
img_back = np.abs(img_back) # 절대값 변환

img_new = np.uint8(img_back) # Float -> Int 변환 (Thresholding 적용)
ret, thresh = cv2.threshold(img_new, 30, 255, cv2.THRESH_BINARY_INV) # Thresholding

plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_back, cmap='gray'), plt.title('FT'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(thresh, cmap='gray'), plt.title('Threshold With FT'), plt.xticks([]), plt.yticks([])
plt.show()
```

---

### Numpy 푸리에 변환 결과 (저주파 제거)

![](images/numpy_fourier_transform_result.png)

---

### Sample Code - OpenCV 푸리에 변환 (고주파 제거)

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/lena_gray.png', 0) # Grayscale 이미지 로드
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) # 푸리에 변환 (OpenCV)
dft_shift = np.fft.fftshift(dft) # 저주파 중앙 이동
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) # Spectrum 계산 (Log Scaling)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2 # 이미지 중심 좌표

d = 30 # 마스크 크기
mask = np.zeros((rows, cols, 2), np.uint8) # 마스크 생성 (검은색)
mask[crow - d:crow + d, ccol - d:ccol + d] = 1 # 마스크 영역 설정 (중심 고주파 제거)

fshift = dft_shift * mask # 마스크 적용 (고주파 제거)
f_ishift = np.fft.ifftshift(fshift) # Inverse Shift
img_back = cv2.idft(f_ishift) # 역 푸리에 변환 (OpenCV)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) # Magnitude 계산

plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray'), plt.title('FT'), plt.xticks([]), plt.yticks([])
plt.show()
```

---

### OpenCV 푸리에 변환 결과 (고주파 제거)

![](images/opencv_fourier_transform_result.png)

---

## 템플릿 매칭 (Template Matching)

---

### Goal

- Template Matching 이용 이미지 검색
- `cv2.matchTemplate()`, `cv2.minMaxLoc()` 함수 사용법 숙지

---

### 개요

- 원본 이미지에서 특정 이미지 (Template) 검색
- `cv2.matchTemplate()` 함수 사용
- Template 이미지 원본 이미지 위 슬라이딩 -> 유사도 비교
- 유사도 Gray 이미지 반환 (매칭 방법 따라 강도 의미 상이)

---

### `cv2.matchTemplate(image, templ, method)`

| Parameter | Type            | 설명                           |
| :-------- | :-------------- | :----------------------------- |
| `image`   | `numpy.ndarray` | 원본 이미지                    |
| `templ`   | `numpy.ndarray` | Template 이미지                |
| `method`  | int             | Matching Method (아래 표 참고) |

**Returns**: Matching 결과 Gray 이미지

---

### Matching Method

| Method 상수            | 설명                                                      |
| :--------------------- | :-------------------------------------------------------- |
| `cv2.TM_SQDIFF`        | 제곱 차이 매칭 (완전 매칭: 0)                             |
| `cv2.TM_SQDIFF_NORMED` | 정규화 제곱 차이 매칭                                     |
| `cv2.TM_CCORR`         | 상관 관계 매칭 (높은 값: 높은 유사도)                     |
| `cv2.TM_CCORR_NORMED`  | 정규화 상관 관계 매칭                                     |
| `cv2.TM_CCOEFF`        | 상관 계수 매칭 (1: 완전 매칭, -1: 최대 불일치, 0: 무관련) |
| `cv2.TM_CCOEFF_NORMED` | 정규화 상관 계수 매칭                                     |

---

### Sample Code - Template Matching Method 비교

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/capture 0.png', 0) # 원본 이미지 (Grayscale)
img2 = img.copy()
template = cv2.imread('images/cap_template.png', 0) # Template 이미지 (Grayscale)

w, h = template.shape[::-1] # Template 이미지 너비, 높이

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] # Matching Method List

for meth in methods:
    img = img2.copy()
    method = eval(meth) # Method 상수 eval

    res = cv2.matchTemplate(img, template, method) # Template Matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # Min/Max 값, 위치

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: # TM_SQDIFF 계열: 최소값이 매칭 위치
        top_left = min_loc
    else: # 나머지 Method: 최대값이 매칭 위치
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 5) # 매칭 영역 사각형 그리기 (White)

    plt.subplot(121), plt.title(meth), plt.imshow(res, cmap='gray'), plt.yticks([]), plt.xticks([]) # Matching 결과 Plot
    plt.subplot(122), plt.imshow(img, cmap='gray') # 매칭 영역 표시 이미지 Plot
    plt.show()
```

---

### Template Matching Method 별 결과

| TM_CCOEFF Method                            | TM_CCOEFF_NORMED Method                            |
| :------------------------------------------ | :------------------------------------------------- |
| ![](images/template_matching_tm_ccoeff.png) | ![](images/template_matching_tm_ccoeff_normed.png) |

| TM_CCORR Method                            | TM_CCORR_NORMED Method                            |
| :----------------------------------------- | :------------------------------------------------ |
| ![](images/template_matching_tm_ccorr.png) | ![](images/template_matching_tm_ccorr_normed.png) |

| TM_SQDIFF Method                            | TM_SQDIFF_NORMED Method                            |
| :------------------------------------------ | :------------------------------------------------- |
| ![](images/template_matching_tm_sqdiff.png) | ![](images/template_matching_tm_sqdiff_normed.png) |

<br>
<br>

**Note**: 좌측: Matching 결과, 우측: 매칭 영역 표시 이미지

---

## 허프 변환 (Hough Transform)

---

### Goal

- 허프 변환 이해
- 허프 변환 이용 이미지 Line 검출
- `cv2.HoughLines()`, `cv2.HoughLinesP()` 함수 사용법 숙지

---

### Theory (이론)

- 이미지 Shape 검색 유명 방법
- 이미지 형태 검색, 누락/깨진 영역 복원
- 직선 방정식 허프 변환 활용

---

### 직선 방정식 허프 변환

- 점 \((x, y)\) 지나는 무수히 많은 직선 방정식: \(y = mx + c\)
- 삼각 함수 변환: \(r = x \cos \theta + y \sin \theta\)
- 각 점 \((x, y)\) 에 대해 \(\theta\) 값 (1~180) 변화 -> 원점 \((0, 0)\) 부터 \((x, y)\) 까지 거리 \(r\) 계산
- \(( \theta, r)\) 2차원 배열 생성 (180개)

![](images/hough_transform_theory.png)
(출처: 위키피디아)

---

### 허프 변환 그래프

- 각 점 2차원 배열 그래프 표현 -> 사인파 그래프
- 3개 방정식 교차점 -> 직선 확률 ↑
- 교차점 \(( \theta, r)\) -> 직선 방정식 계산

![](images/hough_transform_graph.png)

---

### OpenCV 허프 변환

`cv2.HoughLines()` 함수: 허프 변환 구현

```python
cv2.HoughLines(image, rho, theta, threshold)
```

| Parameter   | Type            | 설명                                                               |
| :---------- | :-------------- | :----------------------------------------------------------------- |
| `image`     | `numpy.ndarray` | Input image (8-bit single-channel binary image, Canny Edge 선처리) |
| `rho`       | float           | \(r\) 값 범위 (0~1 실수)                                           |
| `theta`     | float           | \(\theta\) 값 범위 (0~180 정수)                                    |
| `threshold` | int             | 교차점 기준 값 (값 ↓: 선 검출 ↑, 정확도 ↓, 값 ↑: 정확도 ↑, 검출 ↓) |

**Returns**: Line 좌표 정보 (`lines`)

---

### Sample Code - 허프 변환

```python
import cv2
import numpy as np

img = cv2.imread('images/chessboard/frame01.jpg')
img_original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale 변환
edges = cv2.Canny(gray, 50, 150, apertureSize=3) # Canny Edge Detection

lines = cv2.HoughLines(edges, 1, np.pi / 180, 100) # 허프 변환 (직선 검출)

for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # 직선 그리기 (Red)

res = np.vstack((img_original, img)) # 원본 이미지 vs. 허프 변환 결과 이미지 Vertical Stack
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 허프 변환 결과 - Threshold 값 비교

| Threshold: 100                            | Threshold: 130                            |
| :---------------------------------------- | :---------------------------------------- |
| ![](images/hough_transform_thresh100.png) | ![](images/hough_transform_thresh130.png) |

---

### Probabilistic Hough Transform (확률적 허프 변환)

- 허프 변환 최적화 (속도 향상)
- 전체 점 대상 X, 임의 점 이용 직선 검색
- `cv2.HoughLinesP()` 함수 사용
- 장점: 선 시작점, 끝점 반환 -> 화면 표현 용이

```python
cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
```

| Parameter       | Type            | 설명                                                               |
| :-------------- | :-------------- | :----------------------------------------------------------------- |
| `image`         | `numpy.ndarray` | Input image (8-bit single-channel binary image, Canny Edge 선처리) |
| `rho`           | float           | \(r\) 값 범위 (0~1 실수)                                           |
| `theta`         | float           | \(\theta\) 값 범위 (0~180 정수)                                    |
| `threshold`     | int             | 교차점 기준 값 (값 ↓: 선 검출 ↑, 정확도 ↓, 값 ↑: 정확도 ↑, 검출 ↓) |
| `minLineLength` | float           | 선 최소 길이 (값 ↓: 짧은 선 검출 ↑, 값 ↑: 긴 선 검출 ↑)            |
| `maxLineGap`    | float           | 선 최대 간격 (값 ↓: 끊어진 선에 민감, 값 ↑: 끊어진 선 허용)        |

**Returns**: Line 좌표 정보 (`lines`)

---

### Sample Code - 확률적 허프 변환

```python
import cv2
import numpy as np

img = cv2.imread('images/hough_images.jpg')
edges = cv2.Canny(img, 50, 200, apertureSize=3) # Canny Edge Detection
gray = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Grayscale -> BGR (Color Line 그리기)
minLineLength = 100 # 선 최소 길이
maxLineGap = 0 # 선 최대 간격

lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 100, minLineLength, maxLineGap) # 확률적 허프 변환 (직선 검출)
for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3) # 직선 그리기 (Red)

cv2.imshow('img1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 확률적 허프 변환 결과 - MaxLineGap 값 비교

| MinLineLength: 100, MaxLineGap: 10        | MinLineLength: 100, MaxLineGap: 0        |
| :---------------------------------------- | :--------------------------------------- |
| ![](images/probabilistic_hough_gap10.png) | ![](images/probabilistic_hough_gap0.png) |

---

## Hough Circle Transform (허프 원 변환)

---

### Goal

- 허프 변환 이용 이미지 원 검출
- `cv2.HoughCircles()` 함수 사용법 숙지

---

### Theory (이론)

원 방정식:

\[ (x - x*{center})^2 + (y - y*{center})^2 = r^2 \]

- 변수 3개 (\(x*{center}\), \(y*{center}\), \(r\)) -> 비효율적
- Hough Gradient Method: 가장자리 기울기 측정 -> 원 관련 점 확인

---

### `cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)`

| Parameter   | Type            | 설명                                                     |
| :---------- | :-------------- | :------------------------------------------------------- |
| `image`     | `numpy.ndarray` | Input image (8-bit single-channel grayscale image)       |
| `method`    | int             | 검출 방법 (`cv2.HOUGH_GRADIENT` - Hough Gradient Method) |
| `dp`        | float           | 해상도 비율 (`dp=1`: Input Image 동일 해상도)            |
| `minDist`   | float           | 검출 원 중심 최소 거리 (값 ↓: 오검출 ↑, 값 ↑: 미검출 ↑)  |
| `param1`    | float           | Canny Edge 검출기 Parameter 1                            |
| `param2`    | float           | 원 검출 Threshold (값 ↓: 오검출 ↑, 값 ↑: 미검출 ↑)       |
| `minRadius` | int             | 원 최소 반지름                                           |
| `maxRadius` | int             | 원 최대 반지름                                           |

**Returns**: 원 좌표, 반지름 정보 (`circles`)

---

### Sample Code - 허프 원 변환

```python
import cv2
import numpy as np

img = cv2.imread('images/copy.png', 0) # Grayscale 이미지 로드
img = cv2.medianBlur(img, 5) # Median Blur (노이즈 제거)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Grayscale -> BGR (Color 원 그리기)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=25, minRadius=0, maxRadius=0) # 허프 원 변환 (원 검출)

circles = np.uint16(np.around(circles)) # Float -> Int 변환 (정수 좌표)

for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2) # 원 그리기 (Green)
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3) # 중심점 그리기 (Red)

cv2.imshow('img', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 허프 원 변환 결과

![](images/hough_circle_transform_result.png)

---

## Watershed 알고리즘 이용 이미지 분할

---

### Goal

- Watershed 알고리즘 이용 이미지 분할
- `cv2.watershed()` 함수 사용법 숙지

---

### Theory (이론)

- 이미지 Grayscale 변환 -> Pixel 값 (0~255) 높낮이 표현
- 높낮이 -> 지형 높낮이 비유 (높은 부분: 봉우리, 낮은 부분: 계곡)
- 물 붓기 시뮬레이션 -> 물 섞이는 부분 경계선 생성 -> 이미지 분할

![](images/watershed_theory.png)
(출처: CMM Webpage)

---

### Code - Watershed 알고리즘 순서

1. **Grayscale 변환**: 이미지 Grayscale 변환
2. **Binary Image 변환**: Otsu Thresholding 이용 Binary Image 변환
3. **Morphology**: Opening, Closing 이용 노이즈, Hole 제거
4. **Foreground & Background 구분**:
   - Dilation: 배경 확장 (서로 연결 X 영역 -> 배경)
   - Distance Transform: 중심 -> Skeleton Image 생성 -> Thresholding -> 확실한 전경 (Foreground)
5. **Unknow 영역**: 배경 - 전경 영역
6. **Labelling**: 전경 영역 Labelling (객체 구분)
7. **Watershed Algorithm**: Watershed Algorithm 적용, 경계 영역 (값 -1) 붉은색 지정

---

### Sample Code - Watershed 알고리즘

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/water_coins.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale 변환
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Binary Image 변환 (Otsu)

kernel = np.ones((3, 3), np.uint8) # 3x3 Kernel
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) # Opening (노이즈 제거)

sure_bg = cv2.dilate(opening, kernel, iterations=3) # Dilation (배경 확장)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5) # Distance Transform
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0) # Thresholding (전경 추출)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg) # Unknow 영역 계산

ret, markers = cv2.connectedComponents(sure_fg) # Labelling
markers = markers + 1
markers[unknown == 255] = 0 # Unknow 영역 Label 0 지정

markers = cv2.watershed(img, markers) # Watershed Algorithm 적용
img[markers == -1] = [255, 0, 0] # 경계 영역 붉은색 지정

images = [gray, thresh, sure_bg, dist_transform, sure_fg, unknown, markers, img]
titles = ['Gray', 'Binary', 'Sure BG', 'Distance', 'Sure FG', 'Unknow', 'Markers', 'Result']

for i in range(len(images)):
    plt.subplot(2, 4, i + 1), plt.imshow(images[i]), plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()
```

---

### Watershed 알고리즘 결과

![](images/watershed_algorithm_result.png)

---

## k-Nearest Neighbour (k-NN)

---

### Goal

- k-Nearest Neighbour (k-NN) 알고리즘 이해

---

### Theory (이론)

- Machine Learning:
  - 지도 학습 (Supervised Learning): 훈련 데이터 + 정답 제시 -> 미지 데이터 정답 예측
  - 비지도 학습 (Unsupervised Learning): 훈련 데이터만 제시 -> 컴퓨터 스스로 정답 찾기
- k-NN: 지도 학습, 단순 알고리즘

---

### k-NN 알고리즘 원리

- K 값: 판단 범위 단계 (k-최근접 이웃 수)
- 초록색 원 Class 판별 예시:

![](images/knn_theory.png)

- K=3: 빨간색 2개, 파란색 1개 -> 빨간색 Class
- K=7: 빨간색 2개, 파란색 5개 -> 파란색 Class
- 가중치 부여 가능 (가까운 이웃 가중치 ↑)

---

### Sample Code - k-NN Demo

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32) # 훈련 데이터 (25개, 2차원 좌표, 0~100 랜덤 값)
response = np.random.randint(0, 2, (25, 1)).astype(np.float32) # Class Label (0 or 1, 25개)

red = trainData[response.ravel() == 0] # Class 0 (Red) 데이터 추출
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^') # Red 데이터 Plot (삼각형)

blue = trainData[response.ravel() == 1] # Class 1 (Blue) 데이터 추출
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's') # Blue 데이터 Plot (사각형)

newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32) # 새로운 데이터 (1개, 2차원 좌표, 0~100 랜덤 값)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o') # 새로운 데이터 Plot (원)

knn = cv2.ml.KNearest_create() # k-NN 객체 생성
knn.train(trainData, cv2.ml.ROW_SAMPLE, response) # 훈련 데이터 학습
ret, results, neighbours, dist = knn.findNearest(newcomer, 3) # k-NN 알고리즘 (k=3)

print("result : ", results) # 결과 Class
print("neighbours :", neighbours) # 최근접 이웃 Class
print("distance: ", dist) # 거리

plt.show()
```

---

### k-NN Demo 결과

![](images/knn_demo_result.png)

```
result :  [[0.]]
neighbours : [[1.  0.  0.]]
distance:  [[250.  293.  873.]]
```

---

## k-NN 이용 숫자 인식

---

### Goal

- k-NN Machine Learning 알고리즘 이용 손글씨 숫자 인식

---

### 손글씨 숫자 이미지 데이터

![](images/digits_image.png)

- 가로 100개, 세로 50개, 총 5000개 숫자
- 각 숫자: 20x20 해상도

---

### k-NN 숫자 인식 과정

1. **학습하기 (Training)**:
   - 학습 데이터 (손글씨 이미지) 준비
   - 숫자 이미지 배열 저장 (0~9)
   - 배열 Labeling (0~9)
   - Numpy 파일 저장 (학습 데이터)
2. **테스트 (Testing)**:
   - 학습 Numpy 파일 로드
   - 테스트 이미지 (손글씨) 20x20 Resize
   - k-NN 알고리즘 이용 숫자 인식
3. **재학습 (Retraining)**:
   - 테스트 결과 오인식 시 정답 입력
   - 정답 데이터 Numpy 파일 추가 -> 재학습

---

### Sample Code - k-NN 숫자 인식 (학습 & 테스트)

```python
import cv2
import numpy as np
import glob
import sys

FNAME = 'digits.npz' # 학습 데이터 파일 이름

def machineLearning(): # 학습 데이터 생성 함수
    img = cv2.imread('images/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)
    train = x[:, :].reshape(-1, 400).astype(np.float32)
    k = np.arange(10)
    train_labels = np.repeat(k, 500)[:, np.newaxis]
    np.savez(FNAME, train=train, train_labels=train_labels) # 학습 데이터 Numpy 파일 저장

def resize20(pimg): # 테스트 이미지 20x20 Resize 함수
    img = cv2.imread(pimg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayResize = cv2.resize(gray, (20, 20))
    ret, thresh = cv2.threshold(grayResize, 125, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('num', thresh)
    return thresh.reshape(-1, 400).astype(np.float32)

def loadTrainData(fname): # 학습 데이터 로드 함수
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']
    return train, train_labels

def checkDigit(test, train, train_labels): # 숫자 인식 함수 (k-NN)
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)
    return result

if __name__ == '__main__':
    if len(sys.argv) == 1: # Option Check
        print('option : train or test')
        exit(1)
    elif sys.argv[1] == 'train': # Train Option
        machineLearning() # 학습 데이터 생성
    elif sys.argv[1] == 'test': # Test Option
        train, train_labels = loadTrainData(FNAME) # 학습 데이터 로드
        saveNpz = False
        for fname in glob.glob('images/num*.png'): # 테스트 이미지 파일 Loop
            test = resize20(fname) # 테스트 이미지 Resize
            result = checkDigit(test, train, train_labels) # 숫자 인식
            print(result) # 인식 결과 출력
            k = cv2.waitKey(0) # 키 입력 대기
            if k > 47 and k < 58: # 숫자 키 입력 시 재학습
                saveNpz = True
                train = np.append(train, test, axis=0) # 학습 데이터 추가
                newLabel = np.array(int(chr(k))).reshape(-1, 1) # 새로운 Label 생성
                train_labels = np.append(train_labels, newLabel, axis=0) # Label 데이터 추가
        cv2.destroyAllWindows()
        if saveNpz:
            np.savez(FNAME, train=train, train_labels=train_labels) # 재학습 데이터 저장
    else:
        print('unknow option')
```

---

### Sample Code 설명 (함수 단위)

- `machineLearning()`: 5000개 손글씨 이미지 로드 -> 숫자 Cell 분할 -> 배열 저장 -> Label 작업 -> Numpy 파일 저장 (학습 데이터 생성)
- `resize20()`: 직접 쓴 손글씨 이미지 20x20 Resize -> 반환 (테스트 이미지 Resize)
- `loadTrainData()`: 학습 Numpy 파일 로드 (학습 데이터 로드)
- `checkDigit()`: 테스트 데이터, 학습 데이터 이용 k-NN 알고리즘 적용 -> 결과 반환 (숫자 인식)

---

### 학습 실행

```bash
python knn_digit_recognition.py train
```

- `digits.npz` 파일 생성 (학습 결과)

---

### 테스트 실행

```bash
python knn_digit_recognition.py test
```

- Command 창 숫자: 컴퓨터 인식 숫자
- 오른쪽 작은 창: 테스트 이미지 숫자
- 오인식 시 정답 숫자 입력 -> 재학습

---

### 첫 번째 테스트 결과

![](images/knn_digit_recognition_test1.png)

| 손글씨 |  8  |  5  |  4  |  9  |  9  |  1  |  7  |  1  |  6  |  3  |  2  |  0  |
| :----- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 결과1  |  3  |  5  |  1  |  7  |  1  |  1  |  4  |  1  |  5  |  5  |  2  |  4  |

<br>
<br>

**Note**: 12개 중 3개만 정답 (학습 데이터 vs. 마우스 손글씨 차이) -> 재학습 필요

---

### 재학습 후 테스트 결과 향상

| 손글씨 |  8  |  5  |  4  |  9  |  9  |  1  |  7  |  1  |  6  |  3  |  2  |  0  |
| :----- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 결과2  |  3  |  5  |  1  |  7  |  1  |  1  |  4  |  1  |  5  |  3  |  2  |  0  |
| 결과3  |  6  |  5  |  1  |  7  |  1  |  1  |  4  |  1  |  6  |  3  |  2  |  0  |
| 결과4  |  8  |  5  |  1  |  9  |  9  |  1  |  7  |  1  |  6  |  3  |  2  |  0  |
| 결과5  |  8  |  5  |  4  |  9  |  9  |  1  |  7  |  1  |  6  |  3  |  2  |  0  |

<br>
<br>

**Note**: 4번 재학습 후 100% 적중률 달성 (데이터, 재학습 ↑ -> 정확도 ↑)

---

## Demo 준비 1

---

### 사전 준비 작업

- 프로그램 설치:
  - Python 2.7.X
  - OpenCV 2.4.X
  - Python Library: numpy, Pygame, piSerial
- Raspberry Pi:
  - Pi Camera
  - Ultrasonic Sensor 연결

---

### Demo 소스 다운로드

Github: [https://github.com/hamuchiwa/AutoRCCar.git](https://github.com/hamuchiwa/AutoRCCar.git)

```bash
git clone https://github.com/hamuchiwa/AutoRCCar.git
```

<br>
<br>

**Note**: Git 미설치 시 Zip 파일 다운로드 후 압축 해제

---

### 소스 수정 - 아두이노

`arduino/rc_keyboard_control.ino`

```c++
int right_pin = 10;
int left_pin = 9;
int forward_pin = 6;
int reverse_pin = 7;
```

<br>
<br>

**Note**: 아두이노 Pin Number 수정 or RC Car 연결 Pin 번호 맞춤

---

### 소스 수정 - Computer

`computer/xxxx.py` (Python 소스 파일)

```python
self.server_socket.bind(('192.168.1.100', 8000)) # Raspberry Pi Socket IP -> PC IP 변경
self.ser = serial.Serial('/dev/tty.usbmodem1421', 115200, timeout=1) # Serial Port 확인 (아두이노 IDE)
```

---

### 소스 수정 - Raspberry Pi

`raspberryPi/xxxxxx.py` (Python 소스 파일)

- `stream_client.py` (이미지 전송 Client)
- `ultrasonic_client.py` (거리 측정 Client)

```python
# Socket IP -> PC IP 변경
```

---

### 테스트 - Computer & Arduino 연결

1. 아두이노 LED 연결
2. 아두이노 & PC USB 연결
3. `test/rc_control_test.py` 실행

```bash
python rc_control_test.py
```

4. 키보드 상하좌우 버튼 -> LED 점멸 확인

---

### 테스트 - Computer & Raspberry Pi 연결

1. **Computer**: `test/stream_server_test.py` 실행 (Server)

```bash
python stream_server_test.py
```

2. **Raspberry Pi**: `raspberryPi/stream_client.py` 실행 (Client, SSH 접속)

```bash
python stream_client.py
```

3. **Computer**: Camera 이미지 전송 확인

---

### 테스트 - 거리 측정 센서 연결

1. **Computer**: `test/ultrasonic_server_test.py` 실행 (Server)

```bash
python ultrasonic_server_test.py
```

2. **Raspberry Pi**: `raspberryPi/ultrasonic_client.py` 실행 (Client, SSH 접속)

```bash
sudo python ultrasonic_client.py
```

3. **Computer**: 거리 측정 Data (Cm 단위) 출력 확인

---

## Demo 준비 2

---

### Machine Learning 순서

1. 학습 대상 촬영 (Camera) & 정답 Labeling (사람)
2. Labeling 이미지 Machine Learning 학습
3. 학습 결과 XML 파일 생성
4. 실제 적용 시 XML 파일 로드 -> Camera 촬영 내용 학습 데이터 기반 판단

---

### Machine Learning 학습 - 학습 데이터 생성

1. 학습 숫자 모형 준비
2. **Computer**: `computer/collect_training_data.py` 실행

```bash
python collect_training_data.py
```

3. **Raspberry Pi**: `RaspberryPi/stream_client.py` 실행

```bash
python stream_client.py
```

4. **Computer**: 숫자 4 촬영 + 좌측 화살표 버튼 (Labeling) -> 1분 지속
5. 숫자 8 촬영 + 우측 화살표 버튼 -> 1분 지속
6. 숫자 1 촬영 + 위쪽 화살표 버튼 -> 1분 지속
7. 숫자 3 촬영 + 아래쪽 화살표 버튼 -> 1분 지속
8. `q` 키 종료 -> 촬영 이미지, Labeling/버려진 이미지 개수 확인
9. `computer/training_data_temp/test08.npz` -> `computer/training_data` 폴더 복사

---

### Machine Learning 학습 - 학습 실행

**Computer**: `computer/mlp_training.py` 실행

```bash
python mlp_training.py
```

- `computer/mlp_xml/mlp.xml` 파일 생성 (학습 결과)
- 알고리즘: 다층 퍼셉트론 신경망 (Multi Layer Perceptron)

---

### 테스트 - 숫자 모형 인식 및 RC Car 제어

1. **Computer**: `computer/rc_driver.py` 실행

```bash
python rc_driver.py
```

2. **Raspberry Pi**: `raspberryPi/stream_client.py` 실행

```bash
python stream_client.py
```

3. **Camera**: 숫자 모형 촬영 -> LED 점멸 (학습 결과 확인)
4. RC Car Controller 연결 (LED -> Controller) -> 숫자 인식 RC Car 제어 확인

---

## layout: outro

# 감사합니다!

## 질문 있으신가요?

```

```
