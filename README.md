# Numpy-study
Numpy 정리 및 복습
==
## 2021-09-01
numpy 유튜브 정리 완료

- Numpy 특징 학습
- 배열생성 학습(`array()`)
- 배열생성 및 초기화 학습(`zeros()`, `ones()`, `full()`, `eye()`, `tri()`, `empty()`, `_like()`)
- 생성한 값으로 배열 생성 학습(`arange()`, `linspace()`, `logspace()`)
- 랜덤값으로 배열 생성 학습(`random.random()`, `random.randint()`, `random.normal()`, `random.rand()`, `random.randn()`)
- 표준데이터 타입 학습(`bool_`, `int`, `float`등)
- 날짜/시간 배열 생성 학습(`Y`, `M`, `W`, `D`, `datetime64` 등)
- 배열 조회 학습(`array_info()`)
- 인덱싱, 슬라이싱, 펜시 인덱싱 학습
- 배열 값 삽입/수정/삭제/복사 학습(`insert()`, `delete()`, `copy()`)
- 배열 전치 및 축 변경 학습
- 배열 재구조화 학습(`reshape()`, `newaxis()`)
- 배열 크기 변경 학습
- 배열 추가 학습(`append()`)
- 배열 연결 학습(`concatenate()`, `vstack()`, `hstack()`, `dstack()`, `stack()`)
- 배열 분할 학습(`split()`, `vsplit()`, `hsplit()`, `dsplit()`)
- 배열 연산 학습
    - 브로드 캐스팅, 산술연산, 절대값 함수(`absolute()`, `abs()`), 제곱/제곱근 함수(`square`, `sqrt`), 지수와 로그 함수, 삼각함수
    -  집계함수(`sum()`, `cumsum()`, `diff()`, `prod()`, `cumprod()`, `dot()`/`matmul()`, `tensordot()`, `cross()`, `inner()`/`outer()`, `mean()`, `std()`, `var()`, `min()`, `max()`, `argmin()`, `argmax()`, `median()`, `percentile()`, `any()`, `all()`))
    -  비교연산, 불리언 연산자
- 배열 정렬 학습(`partition()`, `sort()`)
- 배열 입출력 학습

---

## 복습 1일차 2021-09-10

- Numpy 특징 복습
- 배열생성 복습(`array()`)
- 배열생성 및 초기화 복습(`zeros()`, `ones()`, `full()`, `eye()`, `tri()`, `empty()`, `_like()`)
- 생성한 값으로 배열 생성 복습(`arange()`, `linspace()`, `logspace()`)
- 랜덤값으로 배열 생성 복습(`random.random()`, `random.randint()`, `random.normal()`, `random.rand()`, `random.randn()`)
- 표준데이터 타입 복습(`bool_`, `int`, `float`등)
- 날짜/시간 배열 생성 복습(`Y`, `M`, `W`, `D`, `datetime64` 등)
- 배열 조회 복습(`array_info()`)
- 인덱싱, 슬라이싱, 펜시 인덱싱 학습
- 배열 값 삽입/수정/삭제/복사 복습(`insert()`, `delete()`, `copy()`)
- 배열 전치 및 축 변경 복습
- 배열 재구조화 복습(`reshape()`, `newaxis()`)
- 배열 크기 변경 복습
- 배열 추가 복습(`append()`)
- 배열 연결 복습(`concatenate()`, `vstack()`, `hstack()`, `dstack()`, `stack()`)
- 배열 분할 복습(`split()`, `vsplit()`, `hsplit()`, `dsplit()`)

-----

## 복습 2일차 2021-09-11

- 배열 연산 학습
    - 브로드 캐스팅, 산술연산, 절대값 함수(`absolute()`, `abs()`), 제곱/제곱근 함수(`square`, `sqrt`), 지수와 로그 함수, 삼각함수
    -  집계함수(`sum()`, `cumsum()`, `diff()`, `prod()`, `cumprod()`, `dot()`/`matmul()`, `tensordot()`, `cross()`, `inner()`/`outer()`, `mean()`, `std()`, `var()`, `min()`, `max()`, `argmin()`, `argmax()`, `median()`, `percentile()`, `any()`, `all()`))
    -  비교연산, 불리언 연산자
- 배열 정렬 학습(`partition()`, `sort()`)
- 배열 입출력 학습

## 빅데이터 수업 2021-09-13

- Numpy 입출력 학습
    - 데모배열 후 저장(`np.save()`, `np.savez()`, `np.load()`, `np.savetxt()`, `np.loadtxt()`
- 배열 연산 학습
    - 산술학습 학습(`np.add()`, `np.subtract()`, `np.multiply()`, `np.divide()`, `np.exp()`, `np.dot()`, `np.sqrt()`, `np.sin()`, `np.cos()`, `np.tan()`, `np.log()`)
    - 비교연산(`equal()`,
    - 집계 함수(`sum()`, `np.sum()`, `min()`, `np.min()`, `max()`, `np.max()`, `np.sum()`, `np.cumsum()`, `np.mean()`, `mean()`, `np.median()`, `np.corrcoef()`, `np.std()`, `std()`)
    - 브로드 캐스팅(다대일 연산), 벡터연산
    - 배열 복사(`np.copy()`)
- 배열 정렬 학습(`partition()`, `sort()`)

-----

## 빅데이터 수업 2021-09-15

- 인덱싱, 슬라이싱 학습(1차원, 2차원, 3차원)
- 불린 인덱싱 학습  
 => 전체 데이터에서 특정 조건에 만족하는 데이터를 추출할 경우 많이 사용된다.  
 numpy의 불린 인덱싱은 각 배열 요소 선택 여부를 True, False로 지정하는 방식이다. => True를 선택한다.  
- 펜시 인덱싱학습(배열에 인덱스 배열을 전달해 요소를 참조하는 방법)
- 배열 변환 학습(`reshape()`, `resize()`, `ravel()`, `append()`, `insert()`, `delete()`)
