# MSE(Measure Square Error)란?

- 평균제곱오차(Mean Squared Error, MSE)는 오차(error)를 제곱한 값의 평균임.
- 오차란 알고리즘이 예측한 값과 실제 정답과의 차이를 의미함.
- 즉, 알고리즘이 정답을 잘 맞출수록 MSE 값은 작아짐. -> 알고리즘의 성능이 좋다는 것을 말함.

![image](https://user-images.githubusercontent.com/42468263/190920354-f54b3137-1a5d-412e-9623-17ee3102f02d.png)

- 수식에서 1/2은 convention으로써 일반적으로 사용되는 값임.
- 해당 값을 사용하는 이유는 MSE를 미분했을 때 제곱이었던 지수가 전체 식에 상수 2로써 곱해지기 때문에 1/2를 곱하여 이를 제거하기 위함.
- 그저 Convention이기 때문에 다른 문헌에는 1/2 값이 없을 수도 있음.

## 특징 - 오차 대비 큰 손실 함수의 증가폭

- MSE는 오차가 커질수록 손실 함수 값이 빠르게 증가함.
- 손실함수의 크기는 오차의 제곱에 비례하여 변함. 그만큼 오차가 커질수록 미분값 역시 커짐.

출처 : [[Deep Learning] 평균제곱오차(MSE) 개념 및 특징](https://heytech.tistory.com/362)
