# [Numerical Prediction Model]
# 선형 회귀 (Linear Regression)


- **선형 회귀 (Linear Regression)**
    - $(x, y)$ Pair를 통해 $y = ax + b$ 에서 기울기와 절편을 찾는 과정
    - Input Data와 Target Data를 통해 기울기와 절편을 찾아 직선의 방정식을 완성

- **경사 하강법 (Gradient Descent)**
    - 모델이 데이터를 잘 표현할 수 있도록 기울기(변화율)을 사용하여 모델을 조금씩 조정하는 최적화 알고리즘
    - 예측값  $y' = wx + b$
        - Input Data를 넣을 때 출력되는, Target Data와 구분되는 값
        1. 무작위로 w와 b를 정한다.
        2. x에서 샘플 하나를 선택하여 y’을 계산
        3. y’과 선택한 샘플의 진짜 y를 비교
        4. y’이 y와 가까워지도록 w, b를 조정
        5. 모든 샘플을 처리할 때까지 다시 2~4 항목을 반복
    - Example
        - 매번 예측값을 구할 때마다, w → w_inc = w + 0.1, b → b_inc = b + 0.1로 수정
        - y_hat_inc = w_inc * x + b_inc
        - $dy'/dw = x, dy'/db = 1$
        - w_new = w + w_rate, b_new = b + b_rate
            - 단점
            1. 수정 폭이 작아 오차값이 클 경우 대처 X
            2. y가 y’보다 크다고 가정했기 때문에, 수정한 결과는 항상 양의 방향으로 커짐
    
- **오차 역전파 (Backpropagation)**
    - 오차값을 통해 w와 b를 update
    - $err = y - y'$
    - $w' = w + \triangle w *err$
    - $b' = b + 1*err$
    - 이 과정을 모든 훈련 데이터를 적용하고, 모든 훈련 데이터를 이용하여 한 단위의 작업을 진행하는 것을 **Epoch**라고 한다.
    
    ```python
    from sklearn.datasets import load_diabetes
    import matplotlib.pyplot as plt
    
    diabetes = load_diabetes()
    x = diabetes.data[:, 2]
    y = diabetes.target
    # Initial value
    w = 1.0
    b = 1.0
    
    # 100 times of Epochs
    for i in range(1, 100):
    		# Backpropagation 
        for x_i, y_i in zip(x, y):
            y_hat = x_i * w + b
            err = y_i - y_hat
            w_rate = x_i
            w = w + w_rate * err
            b = b + 1 * err
    
    plt.scatter(x, y)
    pt1 = (-0.1, -0.1 * w + b)
    pt2 = (0.15, 0.15 * w + b)
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    ```
 
    

# 손실 함수와 경사 하강법의 관계


- **손실 함수 (Loss Function)**
    - 예상한 값과 실제 타겟 값의 차이를 함수로 표현한 것
    - 경사 하강법의 기술적 의미는 ‘어떤 손실 함수가 정의되었을 때, 손실 함수의 값이 최소가 되는 지점을 찾아가는 방법’
    
- **손실 함수와 경사 하강법의 관계**
    - 오차 역전법을 이용한 경사 하강법은 ‘제곱 오차’라는 손실 함수를 미분하는 과정과 같다.
    
- **제곱 오차 함수의 미분**
    
    $SE = \frac{1}{2}(y - \hat{y})^2$
    
    - 가중치에 대한 제곱 오차의 변화율
    
    ${\partial SE \over \partial w} = {\partial \over \partial w}{1\over 2}(y-\hat{y})^2 = (y-\hat{y})(-{\partial \hat{y} \over \partial w}) = -(y-\hat{y})x$
    
    - 제곱 오차의 변화율로 가중치를 Update.
    
    $w = w-{\partial SE \over \partial w} = w + (y -\hat{y})x$
    
    → 빼지 않고 더한 이유는 손실 함수의 낮은 쪽으로 이동하고 싶기 때문
    
    - 절편에 대한 제곱 오차의 변화율
    
    ${\partial SE \over \partial b} = {\partial \over \partial b}{1\over 2}(y-\hat{y})^2 = (y-\hat{y})(-{\partial \hat{y} \over \partial b}) = -(y-\hat{y})1$
    
    - 제곱 오차의 변화율로 절편을 Update
    
    $b =b-{\partial SE \over \partial b} = b + (y -\hat{y})$
    

# Neuron for Linear Regression

```python
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

class Neuron:
    def __init__(self):
        # initial value
        self.w = 1.0
        self.b = 1.0

    # 3개의 입력 신호 (w, x, b)로 y_hat 을 구하는 정방향 계산
    # Calculate predicted values
    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    # 손실 함수에서 그래디언트를 구하여 가중치와 절편을 업데이트하는 역방향 계산
    # Backpropagation
    def backpop(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = self.forpass(x_i)
                err = -(y_i - y_hat)
                w_grad, b_grad = self.backpop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad

neuron = Neuron()
neuron.fit(x, y)

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
pt2 = (0.15, 0.15 * neuron.w + neuron.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
